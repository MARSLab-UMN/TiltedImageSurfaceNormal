from torch.utils.data import DataLoader
from dataset_loader.dataset_loader_scannet import ScannetDataset
from dataset_loader.dataset_loader_nyud import NYUD_Dataset
from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
from dataset_loader.dataset_loader_scannet import Rectified2DOF
from dataset_loader.dataset_loader_scannet import Full2DOF
import numpy as np
import torch


def create_dataset_loader(config):
    # Testing on NYUD
    if config['TEST_DATASET'] == 'nyud':
        train_dataset = NYUD_Dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        test_dataset = NYUD_Dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=4)

        val_dataset = NYUD_Dataset()
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader, val_dataloader

    # Testing on KinectAzure
    if 'kinect_azure' in config['TEST_DATASET']:
        train_dataset = KinectAzureDataset()
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        if config['TEST_DATASET'] == 'kinect_azure_full':
            test_dataset = KinectAzureDataset(usage='test_full')
        elif config['TEST_DATASET'] == 'kinect_azure_gravity_align':
            test_dataset = KinectAzureDataset(usage='test_gravity_align')
        elif config['TEST_DATASET'] == 'kinect_azure_tilted':
            test_dataset = KinectAzureDataset(usage='test_tilted')

        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=16)

        val_dataset = KinectAzureDataset()
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader, val_dataloader

    # ScanNet standard split
    if 'standard' in config['TRAIN_DATASET']:
        train_dataset = ScannetDataset(usage='train', train_test_split=config['TRAIN_DATASET'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                        shuffle=True, num_workers=16, pin_memory=True)

        test_dataset = ScannetDataset(usage='test', train_test_split=config['TEST_DATASET'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=4)

        val_dataset = ScannetDataset(usage='test', train_test_split=config['TEST_DATASET'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader, val_dataloader

    # rectified_2dofa_scannet/framenet
    if 'rectified_2dof' in config['TRAIN_DATASET']:
        train_dataset = Rectified2DOF(usage='train', train_test_split=config['TRAIN_DATASET'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                       shuffle=True, num_workers=16, pin_memory=True)

        test_dataset = Rectified2DOF(usage='test', train_test_split=config['TRAIN_DATASET'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                       shuffle=True, num_workers=16)

        val_dataset = Rectified2DOF(usage='test', train_test_split=config['TRAIN_DATASET'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                       shuffle=True, num_workers=16)

        return train_dataloader, test_dataloader, val_dataloader

    # full_2dof_scannet/framenet
    if 'full_2dof' in config['TRAIN_DATASET']:
        train_dataset = Full2DOF(usage='train', train_test_split=config['TRAIN_DATASET'])
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        test_dataset = Full2DOF(usage='test', train_test_split=config['TRAIN_DATASET'])
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        val_dataset = Full2DOF(usage='test', train_test_split=config['TRAIN_DATASET'])
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        return train_dataloader, test_dataloader, val_dataloader

    return train_dataloader, test_dataloader, val_dataloader


def data_augmentation(sample_batched, config, warper, epoch, iter):
    if 'ga_split' in sample_batched:
        ga_split = sample_batched['ga_split']

    if config['AUGMENTATION'] == 'warp_input':
        gravity_dir = sample_batched['gravity']
        gravity_dir = gravity_dir.cuda()
        alignment_dir = sample_batched['aligned_directions']
        alignment_dir = alignment_dir.cuda()
        sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched,
                                                                     I_g=gravity_dir,
                                                                     I_a=alignment_dir)
    elif config['AUGMENTATION'] == 'random_warp_input':
        num_img_in_batch = sample_batched['mask'].shape[0]
        theta = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 4.0 # pitch augmentation
        phi = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 1.2 # roll augmentation

        gravity_dir = np.vstack((-np.cos(theta)*np.sin(phi),
                                 np.cos(theta)*np.cos(phi),
                                 np.sin(theta)))
        gravity_dir = torch.tensor(gravity_dir.transpose(), dtype=torch.float).view(num_img_in_batch, 3)
        gravity_dir = gravity_dir.cuda()
        Y_dir = torch.tensor([0.0, 1.0, 0.0]).cuda()

        for i in range(0, num_img_in_batch):
            if np.random.ranf() < 0.3: # random warp 2/3 images
                gravity_dir[i, :] = Y_dir

        alignment_dir = Y_dir.repeat(num_img_in_batch, 1)
        sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched, I_g=gravity_dir, I_a=alignment_dir)

    sample_batched['ga_split'] = ga_split

    return sample_batched
