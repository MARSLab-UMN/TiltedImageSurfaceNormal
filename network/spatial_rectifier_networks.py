import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
from warping_2dof_alignment import Warping2DOFAlignment
from network.fpn_architecture import PFPN
from network.fpn_architecture import DFPN
from network.dorn_architecture import DORNBN


class SpatialRectifier(nn.Module):
    def __init__(self, in_channels=3):
        super(SpatialRectifier, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(18)](pretrained=True)

        self.channel = in_channels

        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']
        self.avg_pool = pretrained_model._modules['avgpool']
        self.warping_params_output = nn.Sequential(nn.Linear(512, 128),
                                                   nn.ReLU(True),
                                                   nn.Dropout(),
                                                   nn.Linear(128, 6))

        # clear memory
        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.avg_pool(x4)
        z = self.warping_params_output(torch.flatten(y, 1))
        return z


class SpatialRectifierPFPN(nn.Module):
    def __init__(self, fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]), canonical_view_cnn_ckpt='', sr_cnn_ckpt=''):
        super(SpatialRectifierPFPN, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.canonical_view_cnn = PFPN(backbone='resnet101')

        fc = fc_img
        cc = np.array([160, 120])
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.warp_params_cnn.load_state_dict(torch.load(sr_cnn_ckpt))
        if canonical_view_cnn_ckpt != '':
            self.canonical_view_cnn.load_state_dict(torch.load(canonical_view_cnn_ckpt))

    def forward(self, x):
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6)
        I_a = torch.nn.functional.normalize(v[:, 3:6], dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g, I_a)

        # Step 3: Warp input to be canonical
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')

        # Step 4: Canonical view
        w_y = self.canonical_view_cnn(w_x)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
        y = y.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        n_pred_c = (R_inv.bmm(y)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        return {'I_g': v[:, 0:3], 'I_a': v[:, 3:6], 'n': n_pred_c, 'W_I': w_x, 'W_O': w_y}


class SpatialRectifierDFPN(nn.Module):
    def __init__(self, fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]), canonical_view_cnn_ckpt='', sr_cnn_ckpt=''):
        super(SpatialRectifierDFPN, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.canonical_view_cnn = DFPN(backbone='resnext101')

        fc = fc_img
        cc = np.array([160, 120])
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.warp_params_cnn.load_state_dict(torch.load(sr_cnn_ckpt))
        if canonical_view_cnn_ckpt != '':
            self.canonical_view_cnn.load_state_dict(torch.load(canonical_view_cnn_ckpt))

    def forward(self, x):
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6)
        I_a = torch.nn.functional.normalize(v[:, 3:6], dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g, I_a)

        # Step 3: Warp input to be canonical
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')

        # Step 4: Canonical view
        w_y = self.canonical_view_cnn(w_x)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
        y = y.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        n_pred_c = (R_inv.bmm(y)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Step 7: Join the information between generalized and canonical??

        return {'I_g': v[:, 0:3], 'I_a': v[:, 3:6], 'n': n_pred_c, 'W_I': w_x, 'W_O': w_y}


class SpatialRectifierDORN(nn.Module):
    def __init__(self, fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]), canonical_view_cnn_ckpt='', sr_cnn_ckpt=''):
        super(SpatialRectifierDORN, self).__init__()
        self.warp_params_cnn = SpatialRectifier()
        self.canonical_view_cnn = DORNBN(output_channel=3)

        fc = fc_img
        cc = np.array([160, 120])
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.warp_params_cnn.load_state_dict(torch.load(sr_cnn_ckpt))

        if canonical_view_cnn_ckpt != '':
            self.canonical_view_cnn.load_state_dict(torch.load(canonical_view_cnn_ckpt))

    def forward(self, x):
        # Step 1: Construct warping parameters
        v = self.warp_params_cnn(x)
        I_g = torch.nn.functional.normalize(v[:, 0:3], dim=1, eps=1e-6)
        I_a = torch.nn.functional.normalize(v[:, 3:6], dim=1, eps=1e-6)

        # Step 2: Construct image sampler forward and inverse
        R_inv, img_sampler, inv_img_sampler = self.warp_2dof_alignment.image_sampler_forward_inverse(I_g, I_a)

        # Step 3: Warp input to be canonical
        w_x = torch.nn.functional.grid_sample(x, img_sampler, padding_mode='zeros', mode='bilinear')

        # Step 4: Canonical view
        w_y = self.canonical_view_cnn(w_x)

        # Step 5: Inverse warp the output to be pixel wise with input
        y = torch.nn.functional.grid_sample(w_y, inv_img_sampler, padding_mode='zeros', mode='bilinear')
        y = y.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        n_pred_c = (R_inv.bmm(y)).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        # Step 7: Join the information between generalized and canonical??

        return {'I_g': v[:, 0:3], 'I_a': v[:, 3:6], 'n': n_pred_c, 'W_I': w_x, 'W_O': w_y}


if __name__ == '__main__':
    warp_param_net = SpatialRectifier()
    warp_param_net.cuda()
