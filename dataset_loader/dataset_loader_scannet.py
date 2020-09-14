import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2

class ScannetDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-frames/',
                       usage='test',
                       train_test_split = './data/framenet_train_test_split.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()

        # self.data_info = pickle.load(open('./data/first10scenes_train.pkl', 'rb'))[usage]
        # self.data_info = pickle.load(open('./data/first10_warped_scenes_train.pkl', 'rb'))[usage]
        # self.data_info = pickle.load(open('./data/first10_warped_scenes_train_scene11_warped_test.pkl', 'rb'))[usage]
        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        # self.data_info = pickle.load(open('./data/framenet_train_test_split.pkl', 'rb'))[usage]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        if usage == 'test':
            self.idx = [i for i in range(0, len(self.data_info[0]), 200)]
        self.data_len = len(self.idx) #len(self.data_info[0])
        self.root = root

        #
        # self.intrinsics = [577.591,318.905,578.73,242.684]
        # xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
        # self.mesh_x = cv2.resize((xx - self.intrinsics[1]) / self.intrinsics[0], (320, 240),interpolation=cv2.INTER_NEAREST)
        # self.mesh_y = cv2.resize((yy - self.intrinsics[3]) / self.intrinsics[2], (320, 240),interpolation=cv2.INTER_NEAREST)


    def __getitem__(self, index):
        if self.train_test_plit == './data/framenet_train_test_split.pkl': # get proper path from framenet pkl
            color_info = self.data_info[0][self.idx[index]]
            orient_info = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_info = self.data_info[2][self.idx[index]]

            color_info = self.root + '/' + color_info[27:]
            orient_info = self.root + '/' + orient_info[27:]
            mask_info = self.root + '/' + mask_info[27:]

            # NOTE: To make the pipeline consistent, we need to add this gravity vector
            gravity_tensor = torch.zeros(3, 1)
        else:
            color_info = self.data_info[0][self.idx[index]]
            orient_info = self.data_info[1][self.idx[index]][:-10] + 'normal.png'
            mask_info = self.data_info[2][self.idx[index]]
            gravity_info = orient_info[:-10] + 'gravity.txt'
            gravity_info = gravity_info.replace('scannet-frames', 'scannet-small-frames')
            gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor
        # input_tensor[3, :, :] = self.mesh_x
        # input_tensor[4, :, :] = self.mesh_y

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z, 'gravity': gravity_tensor}

    def __len__(self):
        return self.data_len

class Scannet2DOFAlignmentDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-frames/',
                       usage='test',
                       train_test_split = './data/scannet_gravity_classification.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))

        self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 1)]
        self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 1)]
        self.idx_e3 = [i for i in range(0, len(self.data_info['e3']), 1)]

        if usage == 'test':
            self.idx_e2 = [i for i in range(0, len(self.data_info['e2']), 1000)]
            self.idx_me2 = [i for i in range(0, len(self.data_info['-e2']), 1000)]
            self.idx_e3 = [i for i in range(0, len(self.data_info['e3']), 1000)]

        self.data_len = max((len(self.idx_e2), len(self.idx_me2), len(self.idx_e3)))
        print('data length: ', self.data_len)
        self.root = root

    def __getitem__(self, index):
        data_idx = 0
        data_split = ''
        if np.random.ranf() < 2./3:
            data_idx = self.idx_e2[index % len(self.idx_e2)]
            data_split = 'e2'
            # aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
        else:
            data_idx = self.idx_me2[index % len(self.idx_me2)]
            data_split = '-e2'
        color_info = self.data_info[data_split][data_idx]
        mask_info = color_info.replace('color', 'orient-mask')
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        mask_valid_size = np.sum((orient_mask_tensor > 0))
        while mask_valid_size < 3e4:
            data_split = 'e2'
            index = np.random.randint(0, len(self.idx_e2))
            data_idx = self.idx_e2[index % len(self.idx_e2)]
            color_info = self.data_info[data_split][data_idx]
            mask_info = color_info.replace('color', 'orient-mask')
            orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
            mask_valid_size = np.sum((orient_mask_tensor > 0))
        # elif np.random.ranf() < 11./12:
        #     data_idx = self.idx_me2[index % len(self.idx_me2)]
        #     data_split = '-e2'
        #     # aligned_directions = torch.tensor([0., -1., 0.], dtype=torch.float)
        # else:
        #     data_idx = self.idx_e3[index % len(self.idx_e3)]
        #     data_split = 'e3'
        #     # aligned_directions = torch.tensor([0., 0., 1.], dtype=torch.float)

        orient_info = color_info.replace('color', 'normal')
        gravity_info = color_info.replace('color.png', 'gravity.txt')
        gravity_info = gravity_info.replace('scannet-frames', 'scannet-small-frames')

        if data_split == 'e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        elif data_split == '-e2':
            aligned_directions = torch.tensor([0., 1., 0.], dtype=torch.float)
            gravity_tensor = -torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        elif data_split == 'e3':
            aligned_directions = torch.tensor([0., 0., 1.], dtype=torch.float)
            gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        Z = -self.to_tensor(orient_img) + 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
                'gravity': gravity_tensor, 'aligned_directions': aligned_directions}

    def __len__(self):
        return self.data_len
