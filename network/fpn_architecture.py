import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
# from gravity_warping import GravityAlignedWarping
from warping_2dof_alignment import Warping2DOFAlignment


def weights_init(modules, type='xavier'):
    m = modules
    if isinstance(m, nn.Conv2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight)
        elif type == 'kaiming':  # msra
            torch.nn.init.kaiming_normal_(m.weight)
        else:
            m.weight.data.fill_(1.0)

        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight)
                elif type == 'kaiming':  # msra
                    torch.nn.init.kaiming_normal_(m.weight)
                else:
                    m.weight.data.fill_(1.0)

                if m.bias is not None:
                    m.bias.data.zero_()


class ResNet50Pyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=True):
        super(ResNet50Pyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(50)](pretrained=pretrained)

        # for m in pretrained_model.modules():
        #     print(m)
        # exit()
        self.channel = in_channels

        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        if pretrained:
            weights_init(self.conv1, type='kaiming')
            weights_init(self.layer1[0].conv1, type='kaiming')
            weights_init(self.layer1[0].downsample[0], type='kaiming')
            # weights_init(self.layer3[0].conv2, type='kaiming')
            # weights_init(self.layer3[0].downsample[0], type='kaiming')
            # weights_init(self.layer4[0].conv2, 'kaiming')
            # weights_init(self.layer4[0].downsample[0], 'kaiming')
        else:
            weights_init(self.modules(), type='kaiming')

        if freeze:
            self.freeze()

    def forward(self, x):
        # print(pretrained_model._modules)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # print('conv1:', x.size())

        x = self.maxpool(x)

        # print('pool:', x.size())

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNetPyramids(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=True):
        super(ResNetPyramids, self).__init__()
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(101)](pretrained=pretrained)

        # for m in pretrained_model.modules():
        #     print(m)
        # exit()
        self.channel = in_channels

        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv1_1', nn.Conv2d(self.channel, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('conv1_3', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1_3', nn.BatchNorm2d(128)),
            ('relu1_3', nn.ReLU(inplace=True))
        ]))
        self.bn1 = nn.BatchNorm2d(128)
        # self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer1[0].conv1 = nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.layer1[0].downsample[0] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

        self.layer2 = pretrained_model._modules['layer2']

        self.layer3 = pretrained_model._modules['layer3']
        # self.layer3[0].conv2.stride = (1, 1)
        # self.layer3[0].downsample[0].stride = (1, 1)

        self.layer4 = pretrained_model._modules['layer4']
        # self.layer4[0].conv2.stride = (1, 1)
        # self.layer4[0].downsample[0].stride = (1, 1)

        # clear memory
        del pretrained_model

        if pretrained:
            weights_init(self.conv1, type='kaiming')
            weights_init(self.layer1[0].conv1, type='kaiming')
            weights_init(self.layer1[0].downsample[0], type='kaiming')
            # weights_init(self.layer3[0].conv2, type='kaiming')
            # weights_init(self.layer3[0].downsample[0], type='kaiming')
            # weights_init(self.layer4[0].conv2, 'kaiming')
            # weights_init(self.layer4[0].downsample[0], 'kaiming')
        else:
            weights_init(self.modules(), type='kaiming')

        if freeze:
            self.freeze()

    def forward(self, x_input):
        # print(pretrained_model._modules)

        x = self.conv1(x_input)
        x = self.bn1(x)
        x = self.relu(x)

        # print('conv1:', x.size())

        x = self.maxpool(x)

        # print('pool:', x.size())

        x1 = self.layer1(x)
        # print('layer1 size:', x1.size())
        x2 = self.layer2(x1)
        # print('layer2 size:', x2.size())
        x3 = self.layer3(x2)
        # print('layer3 size:', x3.size())
        x4 = self.layer4(x3)
        # print('layer4 size:', x4.size())
        return {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4}
        # return x4

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


# class ResNetUpsample(nn.Module):
#     def __init__(self, output_size=(240, 320)):
#         super(ResNetUpsample, self).__init__()
#         self.output_size = output_size
#         self.resnet = ResNet(in_channels=6, pretrained=False)
#         self.feature_to_normal = nn.Sequential(
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(2048, 512, 3),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(512, 64, 3),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(p=0.5),
#             nn.Conv2d(64, 3, 3),
#             nn.UpsamplingBilinear2d(size=self.output_size)
#         )
#         # weights_init(self.feature_to_normal, type='xavier')
#
#     def forward(self, x):
#         features = self.resnet(x)
#         return self.feature_to_normal(features)


class ModifiedFPN(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss', use_mask=False):
        super(ModifiedFPN, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.use_mask = use_mask
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        if 'mix_loss' in training_mode:
            self.feature_concat = nn.Sequential(
                # nn.Conv2d(256, 128, 3, 1, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1),
            )
        else:
            self.feature_concat = nn.Sequential(
                # nn.Dropout2d(0.5),
                # nn.Conv2d(256, 128, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1, 1, 1),
                nn.UpsamplingBilinear2d(size=(240, 320)),
            )

        # weights_init(self.feature_concat, type='xavier')

    def forward(self, x):
        features = self.resnet_pyramids(x)
        if self.use_mask:
            feature_mask = x[:, 0:1] + x[:, 1:2] + x[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
            return y
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
            return y


class ModifiedFPNResNet50(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss'):
        super(ModifiedFPNResNet50, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.resnet_pyramids = ResNet50Pyramids(in_channels=in_channels, pretrained=True)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        if 'mix_loss' in training_mode:
            self.feature_concat = nn.Sequential(
                # nn.Conv2d(256, 128, 3, 1, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1),
            )
        else:
            self.feature_concat = nn.Sequential(
                # nn.Dropout2d(0.5),
                # nn.Conv2d(256, 128, 1),
                # nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1, 1, 1),
                nn.UpsamplingBilinear2d(size=(240, 320)),
            )

        # weights_init(self.feature_concat, type='xavier')

    def forward(self, x):
        features = self.resnet_pyramids(x)
        z1 = self.feature1_upsamping(features['x1'])
        z2 = self.feature2_upsamping(features['x2'])
        z3 = self.feature3_upsamping(features['x3'])
        z4 = self.feature4_upsamping(features['x4'])

        y = self.feature_concat(z1 + z2 + z3 + z4)
        if 'mix_loss' in self.mode:
            # tt_slant = torch.nn.functional.leaky_relu(out[:, 2:3], negative_slope=0.02)
            if 'mix_loss_v7' in self.mode:
                z = torch.cat((y[:, 0:1],
                               torch.nn.functional.hardtanh(y[:, 1:2], min_val=-1.57, max_val=1.57),
                               torch.nn.functional.relu(y[:, 2:3])), dim=1)
            else:
                z = torch.cat((y[:, 0:2], torch.nn.functional.leaky_relu_(y[:, 2:3], negative_slope=0.01)), dim=1)
            return nn.functional.interpolate(z, size=(240, 320), mode='bilinear')
        return y



class FPNWarpInput(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3,
                    training_mode='train_L2_loss',
                    fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]),
                    cc_img=np.array([0.5 * 319.87654, 0.5 * 239.87603]),
                    use_mask=False):
        super(FPNWarpInput, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.use_mask = use_mask

        fc = fc_img
        cc = cc_img
        self.gravity_warper = GravityAlignedWarping(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1],
                                                    warped_direction=np.array([0.0, 1.0, 0.0]))
                                                    # warped_direction=np.array([0.0, 0.866, 0.5]))

        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )

        # weights_init(self.feature_concat, type='xavier')
    def forward(self, x, gravity_tensor):
        gravity_tensor = gravity_tensor.view(gravity_tensor.shape[0], gravity_tensor.shape[1])
        _, x1 = self.gravity_warper.warp_with_gravity(x, gravity_tensor)
        # _, x1 = self.gravity_warper.warp_with_gravity_center_aligned(x, gravity_tensor)
        features = self.resnet_pyramids(x1)
        if self.use_mask:
            feature_mask = x[:, 0:1] + x[:, 1:2] + x[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
        _, z = self.gravity_warper.inverse_warp_normal_image_with_gravity(y, gravity_tensor)
        # _, z = self.gravity_warper.inverse_warp_normal_image_with_gravity_center_aligned(y, gravity_tensor)
        return z, x1, y


class FPNWarpInputMultiDirections(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3,
                    training_mode='train_L2_loss',
                    fc_img=np.array([0.5 * 577.87061, 0.5 * 580.25851]),
                    cc_img=np.array([0.5 * 319.87654, 0.5 * 239.87603]),
                    use_mask=False):
        super(FPNWarpInputMultiDirections, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.use_mask = use_mask

        fc = fc_img
        cc = cc_img
        self.warp_2dof_alignment = Warping2DOFAlignment(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1])

        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )

    def forward(self, x):
        features = self.resnet_pyramids(x)
        if self.use_mask:
            feature_mask = x1[:, 0:1] + x1[:, 1:2] + x1[:, 2:3] > 1e-2
            feature_mask = feature_mask.float().detach()
            feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
            feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
            feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
            feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')

            z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
            z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
            z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
            z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
            y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
        else:
            z1 = self.feature1_upsamping(features['x1'])
            z2 = self.feature2_upsamping(features['x2'])
            z3 = self.feature3_upsamping(features['x3'])
            z4 = self.feature4_upsamping(features['x4'])
            y = self.feature_concat(z1 + z2 + z3 + z4)
        return y

    # def forward(self, x, gravity_tensor, alignment_tensor):
    #     _, x1 = self.warp_2dof_alignment.warp_with_gravity_center_aligned(x, gravity_tensor, alignment_tensor)
    #     features = self.resnet_pyramids(x1)
    #     if self.use_mask:
    #         feature_mask = x1[:, 0:1] + x1[:, 1:2] + x1[:, 2:3] > 1e-2
    #         feature_mask = feature_mask.float().detach()
    #         feature1_mask = nn.functional.interpolate(feature_mask, size=(60, 80), mode='nearest')
    #         feature2_mask = nn.functional.interpolate(feature_mask, size=(30, 40), mode='nearest')
    #         feature3_mask = nn.functional.interpolate(feature_mask, size=(15, 20), mode='nearest')
    #         feature4_mask = nn.functional.interpolate(feature_mask, size=(8, 10), mode='nearest')
    #
    #         z1 = self.feature1_upsamping(features['x1'] * feature1_mask)
    #         z2 = self.feature2_upsamping(features['x2'] * feature2_mask)
    #         z3 = self.feature3_upsamping(features['x3'] * feature3_mask)
    #         z4 = self.feature4_upsamping(features['x4'] * feature4_mask)
    #         y = self.feature_concat((z1 + z2 + z3 + z4) * feature1_mask)
    #     else:
    #         z1 = self.feature1_upsamping(features['x1'])
    #         z2 = self.feature2_upsamping(features['x2'])
    #         z3 = self.feature3_upsamping(features['x3'])
    #         z4 = self.feature4_upsamping(features['x4'])
    #         y = self.feature_concat(z1 + z2 + z3 + z4)
    #     _, z = self.warp_2dof_alignment.inverse_warp_normal_image_with_gravity_center_aligned(y, gravity_tensor, alignment_tensor)
    #     return z, x1, y


class FPNWarpFeatures(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss'):
        super(FPNWarpFeatures, self).__init__()
        self.output_size = output_size
        self.mode = training_mode

        fc = np.array([577.87061, 580.25851]) / 2
        cc = np.array([319.87654, 239.87603]) / 2
        self.warp_feature1 = GravityAlignedWarping(fx=fc[0] / 2**2, fy=fc[1] / 2**2, cx=cc[0] / 2**2, cy=cc[1] / 2**2,
                                                         warped_direction=np.array([0.0, 0.866, 0.5]))
        self.warp_feature2 = GravityAlignedWarping(fx=fc[0] / 2**3, fy=fc[1] / 2**3, cx=cc[0] / 2**3, cy=cc[1] / 2**3,
                                                         warped_direction=np.array([0.0, 0.866, 0.5]))
        self.warp_feature3 = GravityAlignedWarping(fx=fc[0] / 2**4, fy=fc[1] / 2**4, cx=cc[0] / 2**4, cy=cc[1] / 2**4,
                                                         warped_direction=np.array([0.0, 0.866, 0.5]))
        self.warp_feature4 = GravityAlignedWarping(fx=fc[0] / 2**5, fy=fc[1] / 2**5, cx=cc[0] / 2**5, cy=cc[1] / 2**5,
                                                         warped_direction=np.array([0.0, 0.866, 0.5]))
        self.warp_output = GravityAlignedWarping(fx=fc[0], fy=fc[1], cx=cc[0], cy=cc[1],
                                                   warped_direction=np.array([0.0, 0.866, 0.5]))

        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.feature1_upsamping = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature2_upsamping = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature3_upsamping = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.feature4_upsamping = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(15, 20)),
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(30, 40)),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(size=(60, 80)),
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.feature_concat = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1),
            nn.UpsamplingBilinear2d(size=(240, 320)),
        )

        # weights_init(self.feature_concat, type='xavier')
    def forward(self, x, gravity_tensor):
        gravity_tensor = gravity_tensor.view(gravity_tensor.shape[0], gravity_tensor.shape[1])
        features = self.resnet_pyramids(x)
        _, x1 = self.warp_output.warp_with_gravity(x, gravity_tensor)
        m = x1[:, 0:1] + x1[:, 1:2] + x1[:, 2:3] > 1e-4
        m = m.float().detach()
        m1 = torch.nn.functional.interpolate(m, size=(60, 80))
        m2 = torch.nn.functional.interpolate(m, size=(30, 40))
        m3 = torch.nn.functional.interpolate(m, size=(15, 20))
        m4 = torch.nn.functional.interpolate(m, size=(8, 10))
        _, warped_x1 = self.warp_feature1.warp_with_gravity(features['x1'] * m1, gravity_tensor)
        _, warped_x2 = self.warp_feature2.warp_with_gravity(features['x2'] * m2, gravity_tensor)
        _, warped_x3 = self.warp_feature3.warp_with_gravity(features['x3'] * m3, gravity_tensor)
        _, warped_x4 = self.warp_feature4.warp_with_gravity(features['x4'] * m4, gravity_tensor)

        z1 = self.feature1_upsamping(warped_x1)
        z2 = self.feature2_upsamping(warped_x2)
        z3 = self.feature3_upsamping(warped_x3)
        z4 = self.feature4_upsamping(warped_x4)
        y = self.feature_concat(m1 * (z1 + z2 + z3 + z4))
        _, z = self.warp_output.inverse_warp_normal_image_with_gravity(y * m, gravity_tensor)
        return z, x1, y


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        width = int(out_dim / 2)
        self.conv0 = nn.Conv2d(in_dim, out_dim, 1)
        self.bn0 = nn.BatchNorm2d(out_dim)

        self.conv1 = nn.Conv2d(in_dim, width, 1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_dim, 1)
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv0(x)
        identity = self.bn0(identity)
        identity = self.relu(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out


class ResidualUpsampleBlock(nn.Module):
    def __init__(self, in_dim, out_dim, upsampling_size):
        super(ResidualUpsampleBlock, self).__init__()
        self.res_block1 = ResidualBlock(in_dim, out_dim)
        self.res_block2 = ResidualBlock(out_dim, out_dim)
        if upsampling_size is None:
            self.ups = None
        else:
            self.ups = nn.UpsamplingBilinear2d(size=upsampling_size)

    def forward(self, x):
        x1 = self.res_block1(x)
        out = self.res_block2(x1)
        if self.ups is None:
            return out
        else:
            return self.ups(out)

class HourGlassResNet(nn.Module):
    def __init__(self, output_size=(240, 320), in_channels=3, training_mode='train_L2_loss'):
        super(HourGlassResNet, self).__init__()
        self.output_size = output_size
        self.mode = training_mode
        self.resnet_pyramids = ResNetPyramids(in_channels=in_channels, pretrained=True)
        self.feature43_upsamping = ResidualUpsampleBlock(2048, 1024, upsampling_size=(15, 20))
        self.feature42_upsamping = ResidualUpsampleBlock(1024, 512, upsampling_size=(30, 40))
        self.feature41_upsamping = ResidualUpsampleBlock(512, 256, upsampling_size=(60, 80))

        self.feature32_upsamping = ResidualUpsampleBlock(1024, 512, upsampling_size=(30, 40))
        self.feature31_upsamping = ResidualUpsampleBlock(512, 256, upsampling_size=(60, 80))

        self.feature21_upsamping = ResidualUpsampleBlock(512, 256, upsampling_size=(60, 80))
        self.feature1_upsamping = ResidualUpsampleBlock(256, 128, upsampling_size=None)

        if 'mix_loss' in training_mode:
            self.feature_concat = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1, 1, 1),
            )
        else:
            self.feature_concat = nn.Sequential(
                # nn.Dropout2d(0.5),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.5),
                nn.Conv2d(64, 3, 1, 1, 1),
                nn.UpsamplingBilinear2d(size=(240, 320)),
            )

        # weights_init(self.feature_concat, type='xavier')

    def forward(self, x):
        features = self.resnet_pyramids(x)
        z43 = self.feature43_upsamping(features['x4']) + features['x3']

        z32 = self.feature32_upsamping(features['x3']) # + features['x2']
        z42 = self.feature42_upsamping(z43) + features['x2']

        z41 = self.feature41_upsamping(z42)
        z31 = self.feature31_upsamping(z32)
        z21 = self.feature21_upsamping(features['x2'])

        z1 =  self.feature1_upsamping(features['x1'] + z21 + z31 + z41)

        y = self.feature_concat(z1)
        if 'mix_loss' in self.mode:
            if 'mix_loss_v7' in self.mode:
                z = torch.cat((y[:, 0:1],
                               torch.nn.functional.hardtanh(y[:, 1:2], min_val=-1.57, max_val=1.57),
                               torch.nn.functional.relu(y[:, 2:3])), dim=1)
            else:
                z = torch.cat((y[:, 0:2], torch.nn.functional.leaky_relu_(y[:, 2:3], negative_slope=0.01)), dim=1)
            return nn.functional.interpolate(z, size=(240, 320), mode='bilinear')
        return y
