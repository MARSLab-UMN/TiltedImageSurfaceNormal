import torch
import numpy as np
import skimage.io as sio
import argparse
from torch.utils.data import DataLoader
from network import dorn_architecture, fpn_architecture, stn_fpn
from dataset_loader.dataset_loader_scannet import ScannetDataset
from dataset_loader.dataset_loader_scannet import Scannet2DOFAlignmentDataset
from dataset_loader.dataset_loader_nyud import NYUD_Dataset
from dataset_loader.dataset_loader_kinectazure import KinectAzureDataset
import os
import cv2
from warping_2dof_alignment import Warping2DOFAlignment


def parsing_configurations():
    parser = argparse.ArgumentParser(description='Train/Test surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--rectified_checkpoint_path', type=str, default='')
    parser.add_argument('--sr_checkpoint_path', type=str,
                        default='/mars/mnt/oitstorage/tien_storage/FPN_warping/spatial_rectifier_2nd_trial/model-epoch-00016-iter-24000.ckpt')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_dataset', type=str, default='./data/scannet_standard_train_test_val_split.pkl')
    parser.add_argument('--test_dataset', type=str, default='./data/scannet_standard_train_test_val_split.pkl')
    parser.add_argument('--net_architecture', type=str, default='dorn')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--augmentation', type=str, default='')

    args = parser.parse_args()

    config = {'LOG_FOLDER': args.log_folder,
              'CKPT_PATH': args.checkpoint_path,
              'RECTIFIED_CKPT_PATH': args.rectified_checkpoint_path,
              'SR_CKPT_PATH': args.sr_checkpoint_path,
              'OPERATION': args.operation,
              'BATCH_SIZE': args.batch_size,
              'LEARNING_RATE': args.learning_rate,
              'TRAIN_DATASET': args.train_dataset,
              'TEST_DATASET': args.test_dataset,
              'ARCHITECTURE': args.net_architecture,
              'AUGMENTATION': args.augmentation,
              'OPTIMIZER': args.optimizer}
    return config


def log(str, fp=None):
    if fp is not None:
        fp.write('%s\n' % (str))
        fp.flush()
    print(str)


def check_nan_ckpt(cnn):
    is_nan_flag = False
    for name, param in cnn.named_parameters():
        if torch.sum(torch.isnan(param.data)):
            is_nan_flag = True
            break
    return is_nan_flag


def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-6).view(dir_x.shape[0],1,dir_x.shape[2],dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l


def compute_surface_normal_angle_error(sample_batched, output_pred, mode='evaluate', angle_type='delta'):
    if 'Z' in sample_batched:
        surface_normal_pred = output_pred
        if mode == 'evaluate':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'])
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            return torch.acos(prediction_error) * 180.0 / np.pi

        elif mode == 'train_L2_loss':
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            mask = sample_batched['mask'] > 0
            mask = mask.detach()
            return -torch.sum(prediction_error[mask]), 1.0-torch.mean(prediction_error[mask])

        elif mode == 'train_L2_explicit_normalize':
            surface_normal_pred = Normalize(surface_normal_pred)
            surface_normal_gt = Normalize(sample_batched['Z'])
            prediction_error = torch.sum((surface_normal_pred - surface_normal_gt) ** 2, dim=1)
            mask = sample_batched['mask'] > 0
            return torch.sum(prediction_error[mask]), 0.5*torch.mean(prediction_error[mask])

        elif mode == 'train_AL_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            acos_mask = mask.float() \
                   * (prediction_error.detach() < 0.999).float() * (prediction_error.detach() > -0.999).float()
            acos_mask = acos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask]))
            logging_loss = torch.mean(torch.acos(prediction_error[acos_mask]))
            return optimize_loss, logging_loss

        elif mode == 'train_TAL_loss':
            mask = sample_batched['mask'] > 0
            prediction_error = torch.cosine_similarity(surface_normal_pred, sample_batched['Z'], dim=1, eps=1e-6)
            # Robust acos loss
            acos_mask = mask.float() \
                   * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
            cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0
            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])
            logging_loss = optimize_loss.detach() / (torch.sum(cos_mask) + torch.sum(acos_mask))
            return optimize_loss, logging_loss

        elif mode == 'train_SR_only':
            prediction_error_g = torch.cosine_similarity(surface_normal_pred['I_g'], sample_batched['gravity'],
                                                       dim=1, eps=1e-6)
            prediction_error_a = torch.cosine_similarity(surface_normal_pred['I_a'], sample_batched['aligned_directions'],
                                                       dim=1, eps=1e-6)

            acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
            cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
            acos_mask_g = acos_mask_g > 0.0
            cos_mask_g = cos_mask_g > 0.0

            acos_mask_a = (prediction_error_a.detach() < 0.9999).float() * (prediction_error_a.detach() > 0.0).float()
            cos_mask_a = (prediction_error_a.detach() <= 0.0).float()
            acos_mask_a = acos_mask_a > 0.0
            cos_mask_a = cos_mask_a > 0.0

            optimize_loss = (torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])) \
                            + (torch.sum(torch.acos(prediction_error_a[acos_mask_a])) - torch.sum(prediction_error_a[cos_mask_a]))
            logging_loss = 0.5*(1.0-torch.mean(prediction_error_g) + 1.0-torch.mean(prediction_error_a))
            return optimize_loss, logging_loss

        elif mode == 'train_sr_fpn_full':
            mask = sample_batched['mask'] > 0
            prediction_error_g = torch.cosine_similarity(surface_normal_pred['I_g'], sample_batched['gravity'],
                                                       dim=1, eps=1e-6)
            prediction_error_a = torch.cosine_similarity(surface_normal_pred['I_a'], sample_batched['aligned_directions'],
                                                       dim=1, eps=1e-6)
            prediction_error = torch.cosine_similarity(surface_normal_pred['n'], sample_batched['Z'], dim=1, eps=1e-6)

            # Robust acos loss
            acos_mask = mask.float() \
                        * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
            cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
            acos_mask = acos_mask > 0.0
            cos_mask = cos_mask > 0.0

            acos_mask_g = (prediction_error_g.detach() < 0.9999).float() * (prediction_error_g.detach() > 0.0).float()
            cos_mask_g = (prediction_error_g.detach() <= 0.0).float()
            acos_mask_g = acos_mask_g > 0.0
            cos_mask_g = cos_mask_g > 0.0

            acos_mask_a = (prediction_error_a.detach() < 0.9999).float() * (prediction_error_a.detach() > 0.0).float()
            cos_mask_a = (prediction_error_a.detach() <= 0.0).float()
            acos_mask_a = acos_mask_a > 0.0
            cos_mask_a = cos_mask_a > 0.0

            optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask]) + \
                            76800 * (torch.sum(torch.acos(prediction_error_g[acos_mask_g])) - torch.sum(prediction_error_g[cos_mask_g])) +\
                            76800 * (torch.sum(torch.acos(prediction_error_a[acos_mask_a])) - torch.sum(prediction_error_a[cos_mask_a]))
            logging_loss = 0.5*(1.0-torch.mean(prediction_error_g) + 1.0-torch.mean(prediction_error_a))
            return optimize_loss, logging_loss


total_normal_errors = None


def accumulate_prediction_error(sample_batched, angle_error_prediction):
    global total_normal_errors
    mask = sample_batched['mask'] > 0
    if total_normal_errors is None:
        total_normal_errors = angle_error_prediction[mask].data.cpu().numpy()
    else:
        total_normal_errors = np.concatenate((total_normal_errors, angle_error_prediction[mask].data.cpu().numpy()))


def log_normal_stats(epoch, iter, normal_error_in_angle, fp=None):
    log('Epoch %d, Iter %d, Mean %f, Median %f, Rmse %f, 5deg %f, 7.5deg %f, 11.25deg %f, 22.5deg %f, 30deg %f' %
    (epoch, iter, np.average(normal_error_in_angle), np.median(normal_error_in_angle),
     np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
     np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
     np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]), fp)

    print('%f %f %f %f %f %f %f %f' %
        (np.average(normal_error_in_angle), np.median(normal_error_in_angle),
        np.sqrt(np.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape),
        np.sum(normal_error_in_angle < 5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 7.5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 11.25) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 22.5) / normal_error_in_angle.shape[0],
        np.sum(normal_error_in_angle < 30) / normal_error_in_angle.shape[0]))


# TODO: clean up this dataset_loader?
def create_dataset_loader(config):
    # Right now nyud only used for testing
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

    if 'kinect_azure' in config['TEST_DATASET']:
        train_dataset = KinectAzureDataset()
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                        shuffle=True, num_workers=16, pin_memory=True)

        if config['TEST_DATASET'] == 'kinect_azure_full':
            test_dataset = KinectAzureDataset(usage='test_full')
        elif config['TEST_DATASET'] == 'kinect_azure_biased_viewing_directions':
            test_dataset = KinectAzureDataset(usage='test_biased_viewing_directions')
        elif config['TEST_DATASET'] == 'kinect_azure_unseen_viewing_directions':
            test_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')

        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=16)

        val_dataset = KinectAzureDataset()
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader, val_dataloader

    if config['TRAIN_DATASET'] == 'scannet_2dof_alignment':
        train_dataset = Scannet2DOFAlignmentDataset(usage='train')
        train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                                      shuffle=True, num_workers=16, pin_memory=True)

        test_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')
        test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
                                     shuffle=False, num_workers=16)

        # test_dataset = Scannet2DOFAlignmentDataset(usage='test')
        # test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
        #                               shuffle=False, num_workers=16)

        val_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')
        val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                                    shuffle=False, num_workers=16)
        return train_dataloader, test_dataloader, val_dataloader

    # TODO: remove?
    # if config['AUGMENTATION'] == 'random_warp_input':
    #     train_dataset = ScannetDataset(usage='train', train_test_split=config['TRAIN_DATASET'])
    #     train_dataloader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
    #                                     shuffle=True, num_workers=16, pin_memory=True)
    #
    #     test_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')
    #     test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
    #                                  shuffle=False, num_workers=16)
    #
    #     # test_dataset = Scannet2DOFAlignmentDataset(usage='test')
    #     # test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'],
    #     #                              shuffle=False, num_workers=16)
    #
    #     val_dataset = KinectAzureDataset(usage='test_unseen_viewing_directions')
    #     val_dataloader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
    #                                 shuffle=False, num_workers=4)
    #     return train_dataloader, test_dataloader, val_dataloader

    # Standard train/test split on ScanNet
    if config['TEST_DATASET'] == 'scannet_standard':
        config['TEST_DATASET'] = './data/scannet_standard_train_test_val_split.pkl'
    if config['TRAIN_DATASET'] == 'scannet_standard':
        config['TRAIN_DATASET'] = './data/scannet_standard_train_test_val_split.pkl'

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


def create_network(config):
    if config['ARCHITECTURE'] == 'dorn':
        cnn = dorn_architecture.DORN(output_channel=3, training_mode=config['OPERATION'])
    elif config['ARCHITECTURE'] == 'dorn_batchnorm':
        cnn = dorn_architecture.DORNBN(output_channel=3, training_mode=config['OPERATION'])
    elif config['ARCHITECTURE'] == 'pfpn':
        cnn = fpn_architecture.PFPN(in_channels=3, training_mode=config['OPERATION'], backbone='resnet101')
    elif config['ARCHITECTURE'] == 'dfpn':
        cnn = fpn_architecture.DFPN(backbone='resnext101')
    elif config['ARCHITECTURE'] == 'spatial_rectifier':
        cnn = stn_fpn.SpatialRectifier()
    elif config['ARCHITECTURE'] == 'sr_pfpn':
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = stn_fpn.SpatialRectifierPFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = stn_fpn.SpatialRectifierPFPN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dfpn':
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = stn_fpn.SpatialRectifierDFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = stn_fpn.SpatialRectifierDFPN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dorn':
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = stn_fpn.SpatialRectifierDORN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = stn_fpn.SpatialRectifierDORN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])

    cnn = cnn.cuda()

    return cnn


_saving_indices = 0


def forward_cnn(sample_batched, cnn, config):
    if config['ARCHITECTURE'] == 'spatial_rectifier':
        v = cnn(sample_batched['image'])
        output_prediction = {'I_g': v[:, 0:3], 'I_a': v[:, 3:6]}

    elif config['ARCHITECTURE'] == 'sr_dfpn' or \
            config['ARCHITECTURE'] == 'sr_pfpn' or \
            config['ARCHITECTURE'] == 'sr_dorn':
        output_prediction = cnn(sample_batched['image'])
        if config['OPERATION'] == 'evaluate':
            return output_prediction['n']
        else:
            return output_prediction

    else:
        output_prediction = cnn(sample_batched['image'])

    return output_prediction


def modify_inputs(sample_batched, config, warper, epoch, iter):
    if config['AUGMENTATION'] == 'crop_input':
        num_img_in_batch = sample_batched['mask'].shape[0]
        theta = (np.random.ranf(num_img_in_batch) - 0.5)* np.pi / 4.0
        phi = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 2.5
        gravity_dir = np.vstack((-np.cos(theta)*np.sin(phi),
                                 np.cos(theta)*np.cos(phi),
                                 np.sin(theta)))
        gravity_dir = torch.tensor(gravity_dir.transpose(), dtype=torch.float).view(num_img_in_batch, 3)
        gravity_dir = gravity_dir.cuda()
        Y_dir = torch.tensor([0.0, 1.0, 0.0]).cuda()

        for i in range(0, num_img_in_batch):
            if np.random.ranf() < 0.5: # random warp 2/3 images
                gravity_dir[i, :] = Y_dir

        alignment_dir = Y_dir.repeat(num_img_in_batch, 1)
        new_sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched,
                                                                         I_g=gravity_dir,
                                                                         I_a=alignment_dir)
        sample_batched['mask'] = new_sample_batched['mask']
        sample_batched['image'] = sample_batched['image'] * new_sample_batched['visible_mask'].\
                                                                view(sample_batched['image'].shape[0],
                                                                     1,
                                                                     sample_batched['image'].shape[2],
                                                                     sample_batched['image'].shape[3])
    elif config['AUGMENTATION'] == 'scale_crop_input':
        num_img_in_batch = sample_batched['mask'].shape[0]
        theta = (np.random.ranf(num_img_in_batch) - 0.5)* np.pi / 3.0
        phi = (np.random.ranf(num_img_in_batch) - 0.5) * np.pi / 3.0
        gravity_dir = np.vstack((np.sin(theta)*np.cos(phi),
                                 np.cos(theta),
                                 np.sin(theta)*np.sin(phi)))
        gravity_dir = torch.tensor(gravity_dir.transpose(), dtype=torch.float).view(num_img_in_batch, 3)
        gravity_dir = gravity_dir.cuda()
        canonical_mask = torch.ones(1, 3, 240, 320).cuda()
        for i in range(0, num_img_in_batch):
            if np.random.ranf() > 0.3: # random warp 2/3 images
                _, warp_mask = warper.warp_with_gravity(canonical_mask, gravity_dir[i, :].view(1, 3))
                warp_mask = warp_mask > 0
                warp_mask = warp_mask.float()
                sample_batched['mask'][i:i+1, :, :] = sample_batched['mask'][i:i+1, :, :] * warp_mask[:, 0]
                sample_batched['image'][i:i+1, :, :, :] = sample_batched['image'][i:i+1, :, :, :] * warp_mask
            scale_factor = np.random.ranf()
            if scale_factor > 0.5:
                scale_factor *= 1.5
                sample_batched['mask'][i:i + 1, :, :] = warper.scale_features(sample_batched['mask'][i:i + 1, :, :], scale_factor)
                sample_batched['image'][i:i + 1, :, :, :] = warper.scale_features(sample_batched['image'][i:i + 1, :, :, :], scale_factor)
                sample_batched['Z'][i:i + 1, :, :, :] = warper.scale_features(sample_batched['Z'][i:i + 1, :, :, :], scale_factor)

    elif config['AUGMENTATION'] == 'warp_input':
        gravity_dir = sample_batched['gravity']
        gravity_dir = gravity_dir.cuda()
        alignment_dir = sample_batched['aligned_directions']
        alignment_dir = alignment_dir.cuda()
        sample_batched = warper.warp_all_with_gravity_center_aligned(sample_batched,
                                                                     I_g=gravity_dir,
                                                                     I_a=alignment_dir)
    elif config['AUGMENTATION'] == 'random_warp_input':
        # global _saving_indices
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

    return sample_batched


if __name__ == '__main__':
    # Step 0. Debugging global parameters
    # _saving_indices = 0

    # Step 1. Configuration file
    config = parsing_configurations()

    # Create logger file
    training_loss_file = None
    evaluate_stat_file = None
    if config['LOG_FOLDER'] != '':
        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])
        training_loss_file = open(config['LOG_FOLDER'] + '/training_loss.txt', 'w')
        evaluate_stat_file = open(config['LOG_FOLDER'] + '/evaluate_stat.txt', 'w')
    log(config, training_loss_file)
    log(config, evaluate_stat_file)

    # Step 2. Create dataset loader
    train_dataloader, test_dataloader, val_dataloader = create_dataset_loader(config)

    # Step 3. Create cnn
    cnn = create_network(config)

    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))

    # Step 4. Create optimizer
    optimizer = None
    if 'train' in config['OPERATION']:
        if config['OPTIMIZER'] == 'adam':
            optimizer = torch.optim.Adam(cnn.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999))
        else:
            raise Exception('Optimizer not implemented!')

    # Step 5. Create warper input:
    warper = Warping2DOFAlignment()

    # Step 6. Learning loop
    if 'train' in config['OPERATION']:
        last_ckpt = {'epoch': None, 'iter': None}
        for epoch in range(0, 20):
            for iter, sample_batched in enumerate(train_dataloader):
                cnn.train()

                sample_batched = {data_key: sample_batched[data_key].cuda() for data_key in sample_batched}

                if config['AUGMENTATION'] != '':
                    sample_batched = modify_inputs(sample_batched, config, warper, epoch, iter)

                # zero the parameter gradients
                optimizer.zero_grad()

                # Step 6b: Forward pass
                output_prediction = forward_cnn(sample_batched, cnn, config)

                # Step 6c: Compute loss
                losses, logging_losses = compute_surface_normal_angle_error(sample_batched,
                                                                            output_prediction,
                                                                            mode=config['OPERATION'],
                                                                            angle_type='delta')
                # Step 6d: Backward pass and update
                losses.backward()
                optimizer.step()

                # Step 6e. Print loss value
                if iter % 60 == 0:
                    log('Epoch %d, Iter %d, Loss %.4f' % (epoch, iter, logging_losses), training_loss_file)

                # Step 6f. Print robust evaluation stats
                if epoch == 0:
                    if iter % 600 == 0 and iter > 0:
                        # Reload closest checkpoint if hit nan
                        if check_nan_ckpt(cnn):
                            if iter > 6000:
                                iter = int((np.ceil(iter / 6000) - 1) * 6000)
                            else:
                                exit()

                            print('Nan ckpt detected, reset optimizer and reload ckpt ep=', epoch, ', iter=', iter)
                            cnn.load_state_dict(torch.load(config['LOG_FOLDER'] + '/model-epoch-%05d-iter-%05d.ckpt' % (epoch, iter)))
                            optimizer = torch.optim.Adam(cnn.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999))

                        evaluation_mode = 'evaluate' + config['OPERATION'][len('train'):] if 'mix_loss' in config['OPERATION'] else 'evaluate'
                        total_normal_errors = None
                        with torch.no_grad():
                            print('<EVALUATION MODE:', evaluation_mode, '>')
                            cnn.eval()
                            for _, eval_batch in enumerate(test_dataloader): # TODO: val_dataloader
                                eval_batch = {data_key: eval_batch[data_key].cuda() for data_key in eval_batch}
                                if config['AUGMENTATION'] == 'warp_input':
                                    eval_batch = modify_inputs(eval_batch, config, warper, epoch, iter)

                                output_prediction = forward_cnn(eval_batch, cnn, config)

                                if 'sr' in config['ARCHITECTURE']:
                                    surfacenormal_pred = output_prediction['n']
                                else:
                                    surfacenormal_pred = output_prediction
                                angle_error_prediction = compute_surface_normal_angle_error(eval_batch,
                                                                                            surfacenormal_pred,
                                                                                            mode=evaluation_mode,
                                                                                            angle_type='delta')
                                accumulate_prediction_error(eval_batch, angle_error_prediction)
                            log_normal_stats(epoch, iter, total_normal_errors, evaluate_stat_file)
                else:
                    if iter % 600 == 0:
                        # Reload closest checkpoint if hit nan
                        if check_nan_ckpt(cnn):
                            if iter > 6000:
                                iter = int((np.ceil(iter/6000)-1)*6000)
                            else:
                                if epoch == 0:
                                    exit()
                                else:
                                    epoch -= 1
                                    iter = int((np.ceil(len(train_dataloader)/6000)-1)*6000)

                            print('Nan ckpt detected, reset optimizer and reload ckpt ep=', epoch, ', iter=', iter)
                            cnn.load_state_dict(torch.load(config['LOG_FOLDER'] + '/model-epoch-%05d-iter-%05d.ckpt' % (epoch, iter)))
                            optimizer = torch.optim.Adam(cnn.parameters(), lr=config['LEARNING_RATE'], betas=(0.9, 0.999))

                        evaluation_mode = 'evaluate' + config['OPERATION'][len('train'):] if 'mix_loss' in config['OPERATION'] else 'evaluate'
                        total_normal_errors = None

                        with torch.no_grad():
                            print('<EVALUATION MODE:', evaluation_mode, '>')
                            cnn.eval()
                            for _, eval_batch in enumerate(test_dataloader): # TODO: val_dataloader
                                eval_batch = {data_key: eval_batch[data_key].cuda() for data_key in eval_batch}
                                if config['AUGMENTATION'] == 'warp_input':
                                    eval_batch = modify_inputs(eval_batch, config, warper, epoch, iter)
                                output_prediction = forward_cnn(eval_batch, cnn, config)
                                if 'sr' in config['ARCHITECTURE']:
                                    surfacenormal_pred = output_prediction['n']
                                else:
                                    surfacenormal_pred = output_prediction
                                angle_error_prediction = compute_surface_normal_angle_error(eval_batch,
                                                                                            surfacenormal_pred,
                                                                                            mode=evaluation_mode,
                                                                                            angle_type='delta')
                                accumulate_prediction_error(eval_batch, angle_error_prediction)
                            log_normal_stats(epoch, iter, total_normal_errors, evaluate_stat_file)

                # Step 6g. Save checkpoints into file
                if iter % 6000 == 0:
                    path = config['LOG_FOLDER'] + '/model-epoch-%05d-iter-%05d.ckpt' % (epoch, iter)
                    torch.save(cnn.state_dict(), path)
    else:
        cnn.eval()
        total_normal_errors = None
        with torch.no_grad():
            for iter, sample_batched in enumerate(test_dataloader):
                sample_batched = {data_key:sample_batched[data_key].cuda() for data_key in sample_batched}
                output_prediction = forward_cnn(sample_batched, cnn, config)
                angle_error_prediction = compute_surface_normal_angle_error(sample_batched, output_prediction,
                                                                            mode=config['OPERATION'], angle_type='delta')
                accumulate_prediction_error(sample_batched, angle_error_prediction)

        # TOTAL error
        print('NORMAL ERROR STATS: ')
        log_normal_stats(0, 0, total_normal_errors)
