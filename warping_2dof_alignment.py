import torch
import numpy as np


class Warping2DOFAlignment:
    def __init__(self, fx=577.87061*0.5, fy=577.87061*0.5, cx=319.87654*0.5, cy=239.87603*0.5):
        self.device = 'cuda:0'
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.W = np.ceil(2 * cx).astype(int)
        self.H = np.ceil(2 * cy).astype(int)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.K)
        self.YY, self.XX = np.meshgrid(np.linspace(0, self.H - 1, self.H), np.linspace(0, self.W - 1, self.W))
        self.corners_points = np.array([[0, 0, 1], [self.W - 1, 0, 1], [0, self.H - 1, 1], [self.W - 1, self.H - 1, 1]]).transpose()
        self.K = torch.tensor(self.K, dtype=torch.float).to(self.device)
        self.K_inv = torch.tensor(self.K_inv, dtype=torch.float).to(self.device)
        self.YY = torch.tensor(self.YY, dtype=torch.float).to(self.device)
        self.XX = torch.tensor(self.XX, dtype=torch.float).to(self.device)
        self.corners_points = torch.tensor(self.corners_points, dtype=torch.float).to(self.device)
        self.I3 = torch.tensor(np.identity(3), dtype=torch.float).to(self.device)

    def _skewsymm(self, x):
        if x.shape[0] == 1:
            return torch.tensor([[0.0, -x[0, 2], x[0, 1]], [x[0, 2], 0.0, -x[0, 0]], [-x[0, 1], x[0, 0], 0.0]], dtype=torch.float).to(
                self.device)
        else:
            return torch.tensor([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]], dtype=torch.float).to(
                self.device)

    def _build_homography(self, I_g, I_a):
        Cg_R_C = torch.zeros((I_g.shape[0], 3, 3), device=I_g.device, dtype=torch.float)
        skewsymm_I_a = torch.zeros((I_g.shape[0], 3, 3), device=I_g.device, dtype=torch.float)
        I_g = I_g.view(I_g.shape[0], 3, 1)
        I_a = I_a.view(I_a.shape[0], 1, 3)
        for i in range(I_g.shape[0]):
            skewsymm_I_a[i] = -self._skewsymm(I_a[i].view(-1))
        Cg_q_C = skewsymm_I_a @ I_g
        dot_Ig_warped_direction = I_a @ I_g
        norm_Cg_q_C = Cg_q_C.clone().norm(dim=1)
        Cg_q4_C = torch.cuda.FloatTensor(I_a.shape[0], 1).fill_(0)

        for i in range(Cg_R_C.shape[0]):
            Cg_q4_C[i] = torch.cos(0.5 * torch.atan2(norm_Cg_q_C[i, 0], dot_Ig_warped_direction[i, 0, 0]))
            if norm_Cg_q_C[i] < 1e-4 or Cg_q4_C[i].abs() < 1e-4:
                Cg_R_C[i] = self.I3
            Cg_q_C[i] = Cg_q_C[i].div((2.0 * Cg_q4_C[i]))
            skewsymm_Cg_q_C = self._skewsymm(Cg_q_C[i].view(-1))
            Cg_R_C[i] = self.I3 + 2. * Cg_q4_C[i] * skewsymm_Cg_q_C + \
                                      2. * skewsymm_Cg_q_C @ skewsymm_Cg_q_C
        Cg_H_C = self.K @ Cg_R_C @ self.K_inv
        C_R_Cg = Cg_R_C.permute(0, 2, 1)
        Cg_H_C_inv = self.K @ C_R_Cg @ self.K_inv
        return Cg_H_C, Cg_R_C, Cg_H_C_inv

    def warp_with_gravity_center_aligned(self, x, I_g, I_a, interp_mode='bilinear'):
        flag_fix_return = False
        if len(x.shape) == 3:
            x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
            flag_fix_return = True

        Cg_H_C, _, Cg_H_C_inv = self._build_homography(I_g, I_a)
        Cg_H_C = Cg_H_C.type(torch.float)
        Cg_H_C_inv = Cg_H_C_inv.type(torch.float)

        grid_sampler = torch.cuda.FloatTensor(x.shape[0], self.H, self.W, 2).fill_(0)
        Cg_corners_points = torch.cuda.FloatTensor(x.shape[0], 3, 4).fill_(0)
        Cg_corners_points_projection = torch.cuda.FloatTensor(x.shape[0], 2, 4).fill_(0)
        C1p_projection = torch.cuda.FloatTensor(x.shape[0], 2, 240*320).fill_(0)
        C1p = torch.cuda.FloatTensor(x.shape[0], 3, 240 * 320).fill_(0)
        assert x.shape[0] == I_g.shape[0]
        for i in range(x.shape[0]):
            Cg_corners_points[i] = Cg_H_C[i] @ self.corners_points # Need proper broadcast w/ batchsize as input
            Cg_corners_points_projection[i] = Cg_corners_points[i, 0:2].clone() / Cg_corners_points[i, 2].clone()
            px_max = torch.max(Cg_corners_points_projection[i, 0].clone())
            px_min = torch.min(Cg_corners_points_projection[i, 0].clone())
            py_max = torch.max(Cg_corners_points_projection[i, 1].clone())
            py_min = torch.min(Cg_corners_points_projection[i, 1].clone())

            h_max = py_max.clone() - py_min.clone()
            w_max = px_max.clone() - px_min.clone()

            if w_max > 4 * h_max / 3:
                kw = self.W / w_max.clone()
                kh = self.H / (3 * w_max.clone() / 4)
            else:
                kh = self.H / (h_max.clone())
                kw = self.W / (4 * h_max.clone() / 3)

            C1p[i] = Cg_H_C_inv[i] @ torch.reshape(torch.cat((1./kw.clone() * (self.XX) + px_min.clone(),
                                                              1./kh.clone() * (self.YY) + py_min.clone(),
                                                              torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))

            C1p_projection[i, 0] = C1p[i, 0].clone() / C1p[i, 2].clone()
            C1p_projection[i, 1] = C1p[i, 1].clone() / C1p[i, 2].clone()

            grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (C1p_projection[i, 0] - self.cx), (self.W, self.H)).t()
            grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (C1p_projection[i, 1] - self.cy), (self.W, self.H)).t()

        y = torch.nn.functional.grid_sample(x, grid_sampler, padding_mode='zeros', mode=interp_mode)
        if flag_fix_return:
            return Cg_H_C, y.view(x.shape[0], x.shape[2], x.shape[3])
        else:
            return Cg_H_C, y

    def image_sampler_forward_inverse(self, I_g, I_a):
        Cg_H_C, Cg_R_C, Cg_H_C_inv = self._build_homography(I_g, I_a)
        Cg_H_C = Cg_H_C.type(torch.float)
        Cg_H_C_inv = Cg_H_C_inv.type(torch.float)
        C_R_Cg = Cg_R_C.permute(0, 2, 1)

        grid_sampler = torch.cuda.FloatTensor(I_g.shape[0], self.H, self.W, 2).fill_(0)
        inv_grid_sampler = torch.cuda.FloatTensor(I_g.shape[0], self.H, self.W, 2).fill_(0)
        C_R_Cg_ret = torch.zeros_like(C_R_Cg)

        for i in range(I_g.shape[0]):
            Cg_corners_points = Cg_H_C[i] @ self.corners_points # Need proper broadcast w/ batchsize as input
            Cg_corners_points_projection = Cg_corners_points[0:2].clone() / Cg_corners_points[2].clone()

            px_max = torch.max(Cg_corners_points_projection[0])
            px_min = torch.min(Cg_corners_points_projection[0])
            py_max = torch.max(Cg_corners_points_projection[1])
            py_min = torch.min(Cg_corners_points_projection[1])

            h_max = py_max - py_min
            w_max = px_max - px_min
            scale_sigma = w_max.detach() / h_max.detach()
            if scale_sigma < 0.8 or scale_sigma > 2.2:
                C_R_Cg_ret[i] = self.I3
                grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (self.XX - self.cx), (self.W, self.H)).t()
                grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (self.YY - self.cy), (self.W, self.H)).t()
                inv_grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (self.XX - self.cx),
                                                             (self.W, self.H)).t()
                inv_grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (self.YY - self.cy),
                                                             (self.W, self.H)).t()
                continue

            if w_max > 4 * h_max / 3:
                kw = self.W / w_max.clone()
                kh = self.H / (3 * w_max.clone() / 4)
            else:
                kh = self.H / (h_max.clone())
                kw = self.W / (4 * h_max.clone() / 3)

            C1p = Cg_H_C_inv[i] @ torch.reshape(torch.cat((1./kw * (self.XX) + px_min,
                                                           1./kh * (self.YY) + py_min,
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))
            C1p_x = C1p[0].clone() / C1p[2].clone()
            C1p_y = C1p[1].clone() / C1p[2].clone()
            grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (C1p_x - self.cx), (self.W, self.H)).t()
            grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (C1p_y - self.cy), (self.W, self.H)).t()

            inv_C1p = Cg_H_C[i] @ torch.reshape(torch.cat((self.XX,
                                                           self.YY,
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))
            inv_C1p_projection = inv_C1p[0:2, :].clone() / inv_C1p[2, :].clone()
            inv_C1p_x = kw * (inv_C1p_projection[0, :] - px_min)
            inv_C1p_y = kh * (inv_C1p_projection[1, :] - py_min)
            inv_grid_sampler[i, :, :, 0] = torch.reshape(1. / (self.W / 2) * (inv_C1p_x - self.cx), (self.W, self.H)).t()
            inv_grid_sampler[i, :, :, 1] = torch.reshape(1. / (self.H / 2) * (inv_C1p_y - self.cy), (self.W, self.H)).t()
            C_R_Cg_ret[i] = C_R_Cg[i]
        # y = torch.nn.functional.grid_sample(x, grid_sampler, padding_mode='zeros', mode=interp_mode)
        return C_R_Cg_ret, grid_sampler, inv_grid_sampler

    def warp_all_with_gravity_center_aligned(self, x, I_g,  I_a):
        x_i = x['image']
        x_m = x['mask']
        x_n = x['Z']
        C_g = x['gravity']
        C_g = C_g.view(C_g.shape[0], 3, 1)
        C_a = x['aligned_directions']
        C_a = C_a.view(C_a.shape[0], 3, 1)
        w_x = {'image': [], 'mask': [], 'gravity': [], 'aligned_directions': [], 'Z':[]}

        x_m = x_m.view(x_m.shape[0], 1, x_m.shape[1], x_m.shape[2])

        Cg_H_C, Cg_R_C, Cg_H_C_inv = self._build_homography(I_g, I_a)
        Cg_H_C = Cg_H_C.type(torch.float)
        Cg_H_C_inv = Cg_H_C_inv.type(torch.float)

        grid_sampler = torch.cuda.FloatTensor(x_i.shape[0], self.H, self.W, 2).fill_(0)
        assert x_i.shape[0] == I_g.shape[0]
        for i in range(x_i.shape[0]):
            Cg_corners_points = Cg_H_C[i] @ self.corners_points # Need proper broadcast w/ batchsize as input
            Cg_corners_points = Cg_corners_points[0:2].clone() / Cg_corners_points[2].clone()
            h_max = torch.max(Cg_corners_points[1, :]) - torch.min(Cg_corners_points[1, :])
            w_max = torch.max(Cg_corners_points[0, :]) - torch.min(Cg_corners_points[0, :])

            if w_max > 4 * h_max / 3:
                kw = self.W / w_max
                kh = self.H / (3 * w_max / 4)
            else:
                kh = self.H / (h_max)
                kw = self.W / (4 * h_max / 3)

            C1p = Cg_H_C_inv[i] @ torch.reshape(torch.cat((1./kw * (self.XX) + torch.min(Cg_corners_points[0, :]),
                                                           1./kh * (self.YY) + torch.min(Cg_corners_points[1, :]),
                                                           torch.ones_like(self.XX)), dim=0), (3, self.W*self.H))
            C1p = C1p.div(C1p[2, :])
            C1p[0, :] = 1. / (self.W / 2) * (C1p[0, :] - self.cx)
            C1p[1, :] = 1. / (self.H / 2) * (C1p[1, :] - self.cy)
            grid_sampler[i, :, :, 0] = torch.reshape(C1p[0, :], (self.W, self.H)).t()
            grid_sampler[i, :, :, 1] = torch.reshape(C1p[1, :], (self.W, self.H)).t()

        y_i = torch.nn.functional.grid_sample(x_i, grid_sampler, padding_mode='zeros', mode='bilinear')
        y_m = torch.nn.functional.grid_sample(x_m, grid_sampler, padding_mode='zeros', mode='nearest')
        y_m = y_m.view(x_m.shape[0], x_m.shape[2], x_m.shape[3])
        y_n = torch.nn.functional.grid_sample(x_n, grid_sampler, padding_mode='zeros', mode='bilinear')
        z_n = Cg_R_C.bmm(y_n.view(x_n.shape[0], x_n.shape[1], x_n.shape[2] * x_n.shape[3]))
        z_n = z_n.view(x_n.shape[0], x_n.shape[1], x_n.shape[2],  x_n.shape[3])
        y_visible = torch.nn.functional.grid_sample(torch.ones_like(x_m), grid_sampler, padding_mode='zeros', mode='bilinear')

        w_x['image'] = y_i
        w_x['mask'] = y_m
        w_x['Z'] = z_n
        w_x['gravity'] = Cg_R_C.bmm(C_g).view(C_g.shape[0], 3)
        w_x['visible_mask'] = y_visible
        w_x['aligned_directions'] = C_a.view(C_a.shape[0], 3) # Cg_R_C.bmm(C_a).view(C_a.shape[0], 3)
        return w_x