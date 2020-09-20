import torch
from network import dorn_architecture, fpn_architecture, spatial_rectifier_networks


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
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = spatial_rectifier_networks.SpatialRectifierPFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = spatial_rectifier_networks.SpatialRectifierPFPN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dfpn':
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = spatial_rectifier_networks.SpatialRectifierDFPN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = spatial_rectifier_networks.SpatialRectifierDFPN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])
    elif config['ARCHITECTURE'] == 'sr_dorn':
        if 'kinect_azure' in config['TEST_DATASET']:
            cnn = spatial_rectifier_networks.SpatialRectifierDORN(sr_cnn_ckpt=config['SR_CKPT_PATH'])
        else:
            cnn = spatial_rectifier_networks.SpatialRectifierDORN(canonical_view_cnn_ckpt=config['RECTIFIED_CKPT_PATH'], sr_cnn_ckpt=config['SR_CKPT_PATH'])

    cnn = cnn.cuda()
    return cnn


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