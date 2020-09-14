import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
import skimage.io as sio
import pickle
import numpy as np
import cv2
import os

class KinectAzureDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/KinectAzure/',
                       usage='test_full',
                       train_test_split = './data/kinect_azure_test_datasets.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.root = root

    def __getitem__(self, index):
        color_info = os.path.join(self.root, self.data_info[0][self.idx[index]])
        orient_info = os.path.join(self.root, self.data_info[1][self.idx[index]])
        mask_info = os.path.join(self.root, self.data_info[2][self.idx[index]])
        gravity_info = color_info.replace('color', 'gravity').replace('png', 'txt')
        # gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        # gravity_tensor[1] = -gravity_tensor[1]
        # gravity_tensor[2] = -gravity_tensor[2]

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

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z} #, 'gravity': gravity_tensor}

    def __len__(self):
        return self.data_len

class KinectAzureDataset2DOFAlignment(Dataset):
    def __init__(self, root='/mars/mnt/dgx/KinectAzure/',
                       usage='test_full',
                       train_test_split = './data/kinect_azure_test_datasets.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split
        self.data_info = pickle.load(open(train_test_split, 'rb'))[usage]

        self.idx = [i for i in range(0, len(self.data_info[0]), 1)]
        self.data_len = len(self.idx)
        self.root = root

    def __getitem__(self, index):
        color_info = os.path.join(self.root, self.data_info[0][self.idx[index]])
        orient_info = os.path.join(self.root, self.data_info[1][self.idx[index]])
        mask_info = color_info.replace('color', 'mask')
        gravity_info = color_info.replace('color', 'gravity').replace('png', 'txt')
        gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        gravity_tensor[1] = -gravity_tensor[1]
        gravity_tensor[2] = -gravity_tensor[2]

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


        psi = gravity_tensor[1]*gravity_tensor[1] + gravity_tensor[2]*gravity_tensor[2]
        if psi < 1e-4:
            # alignment_tensor = torch.tensor([0., torch.cos(pitch_angle), torch.sin(pitch_angle)])
            alignment_tensor = torch.tensor([0., 1., 0.])
            # alignment_tensor = gravity_tensor
        else:
            pitch_angle = torch.atan2(gravity_tensor[2], gravity_tensor[1])
            # print('pitch angle: ', pitch_angle)
            # print('gravity_tensor: ', gravity_tensor)
            if torch.cos(pitch_angle) > 0.5:
                alignment_tensor = torch.tensor([0., 1., 0.])
            else:
                alignment_tensor = torch.tensor([0., torch.cos(pitch_angle), torch.sin(pitch_angle)])
        # print('alignment_tensor: ', alignment_tensor)
        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
                'gravity': gravity_tensor, 'aligned_directions': alignment_tensor}

    def __len__(self):
        return self.data_len

class KinectAzureDistributionMatch(Dataset):
    def __init__(self, root='/mars/mnt/oitstorage/tien_storage/SurfaceNormal_eccv2020/',
                 usage='test_full',
                 train_test_split='./data/kinect_azure_test_datasets.pkl'):
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()
        self.train_test_plit = train_test_split

        self.data_len = 4000
        self.root = root

    def __getitem__(self, index):
        color_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_input.png' % index)
        orient_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_truth.png' % index)
        mask_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_wrong_canonical.png' % index)
        distribution_match_info = os.path.join(self.root, 'script/distribution_match', '%d_roll_opt_results.txt' % index)

        # imgidx=3541
        # color_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_input.png' % imgidx)
        # orient_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_truth.png' % imgidx)
        # mask_info = os.path.join(self.root, 'raw_output_surface_normal_ECCV2020/stn_fpn', '%d_wrong_canonical.png' % imgidx)
        # distribution_match_info = os.path.join(self.root, 'script/distribution_match', '%d_roll_opt_results.txt' % imgidx)

        quaternion_tensor = np.loadtxt(distribution_match_info, dtype=np.float, delimiter=',')
        quaternion_tensor = np.reshape(quaternion_tensor, (6, -1))

        ids = np.where(quaternion_tensor[1, :] > 0.6)
        if len(ids[0]) > 0:
            quaternion_tensor = torch.tensor(quaternion_tensor[2:, ids[0][-1]], dtype=torch.float)
        else:
            quaternion_tensor = torch.tensor([0, 0, 0, 1], dtype=torch.float)


        # quaternion_tensor = torch.tensor(quaternion_tensor[2:, index], dtype=torch.float)


        # quaternion_tensor = torch.tensor([0, 0, np.sin(np.pi/4.0*(index/15-1.0)), np.cos(np.pi/4.0*(index/15-1.0))], dtype=torch.float)

        # Image resize and load
        color_img = cv2.resize(sio.imread(color_info), (320, 240), interpolation=cv2.INTER_CUBIC)
        orient_mask_tensor = cv2.resize(sio.imread(mask_info), (320, 240), interpolation=cv2.INTER_NEAREST)
        orient_img = cv2.resize(sio.imread(orient_info), (320, 240), interpolation=cv2.INTER_NEAREST)

        # To tensor
        color_tensor = self.to_tensor(color_img)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor[:, :, 0] != 128)
        Z = self.to_tensor(orient_img) - 0.5

        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        return {'image': input_tensor, 'mask': orient_mask_tensor, 'Z': Z,
                'quaternion': quaternion_tensor}

    def __len__(self):
        return 4000
