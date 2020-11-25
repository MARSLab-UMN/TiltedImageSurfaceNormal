import torch
import numpy as np
import skimage.io as sio
import argparse
from torch.utils.data import DataLoader
from network import dorn_architecture, fpn_architecture, spatial_rectifier_networks
from dataset_loader.dataset_loader_custom import CustomDataset
import os
import time


def parsing_configurations():
    parser = argparse.ArgumentParser(description='Inference for surface normal estimation')
    parser.add_argument('--log_folder', type=str, default='')
    parser.add_argument('--operation', type=str, default='inference')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--sr_checkpoint_path', type=str, default='./checkpoints/SR_only.ckpt')
    parser.add_argument('--test_dataset', type=str, default='custom folder')
    parser.add_argument('--net_architecture', type=str, default='sr_dfpn')

    args = parser.parse_args()

    config = {'LOG_FOLDER': args.log_folder,
              'CKPT_PATH': args.checkpoint_path,
              'SR_CKPT_PATH': args.sr_checkpoint_path,
              'OPERATION': args.operation,
              'BATCH_SIZE': args.batch_size,
              'TEST_DATASET': args.test_dataset,
              'ARCHITECTURE': args.net_architecture}
    return config


def log(str, fp=None):
    if fp is not None:
        fp.write('%s\n' % (str))
        fp.flush()
    print(str)


def saving_rgb_tensor_to_file(rgb_tensor, path):
    output_rgb_img = np.uint8((rgb_tensor.permute(1, 2, 0).detach().cpu()) * 255)
    sio.imsave(path, output_rgb_img)


def saving_normal_tensor_to_file(normal_tensor, path):
    normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=0)
    output_normal_img = np.uint8((normal_tensor.permute(1, 2, 0).detach().cpu() + 1) * 127.5)
    sio.imsave(path, output_normal_img)


def Normalize(dir_x):
    dir_x_l = torch.sqrt(torch.sum(dir_x ** 2,dim=1) + 1e-6).view(dir_x.shape[0], 1, dir_x.shape[2], dir_x.shape[3])
    dir_x_l = torch.cat([dir_x_l, dir_x_l, dir_x_l], dim=1)
    return dir_x / dir_x_l


def create_dataset_loader(config):
    test_dataset = CustomDataset(dataset_path=config['TEST_DATASET'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=16)

    return test_dataloader


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
        cnn = spatial_rectifier_networks.SpatialRectifier()
    elif config['ARCHITECTURE'] == 'sr_pfpn':
        cnn = spatial_rectifier_networks.SpatialRectifierPFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dfpn':
        cnn = spatial_rectifier_networks.SpatialRectifierDFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dorn':
        cnn = spatial_rectifier_networks.SpatialRectifierDORN(sr_cnn_ckpt=config['SR_CKPT_PATH'])

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

        if config['OPERATION'] == 'inference':
            return output_prediction['n']
        else:
            return output_prediction

    else:
        output_prediction = cnn(sample_batched['image'])

    return output_prediction


if __name__ == '__main__':

    _saving_indices = 0

    # Step 1. Configuration file
    config = parsing_configurations()
    if config['LOG_FOLDER'] != '':
        if not os.path.exists(config['LOG_FOLDER']):
            os.makedirs(config['LOG_FOLDER'])

    # Step 2. Create dataset loader
    test_dataloader = create_dataset_loader(config)

    # Step 3. Create cnn
    cnn = create_network(config)
    if config['CKPT_PATH'] is not '':
        print('Loading checkpoint from %s' % config['CKPT_PATH'])
        cnn.load_state_dict(torch.load(config['CKPT_PATH']))

    counter = 0
    runtime_measurements = []
    with torch.no_grad():
        print('<INFERENCE MODE>')
        cnn.eval()
        for iter, sample_batched in enumerate(test_dataloader):
            print(iter, '/', len(test_dataloader))
            sample_batched = {data_key: sample_batched[data_key].cuda() for data_key in sample_batched}
            torch.cuda.synchronize()
            start_time = time.time()
            output_prediction = forward_cnn(sample_batched, cnn, config)
            torch.cuda.synchronize()
            runtime_measurements.append((time.time() - start_time) / config['BATCH_SIZE'])

            for i in range(output_prediction.shape[0]):
                saving_rgb_tensor_to_file(sample_batched['image'][i],
                                          os.path.join(config['LOG_FOLDER'], 'input_%d.png' % _saving_indices))
                saving_normal_tensor_to_file(output_prediction[i],
                                             os.path.join(config['LOG_FOLDER'],
                                                          'normal_pred_%d.png' % _saving_indices))
                _saving_indices += 1

    print('Median of inference time per image: %.4f (s)' % np.median(np.asarray(runtime_measurements)))