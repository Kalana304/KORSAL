# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo
from torchvision.models.utils import load_state_dict_from_url

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                inplanes, 
                planes, 
                stride = 1, 
                downsample = None, 
                downsample_difference = None,
                final = False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.final = final
        #Difference operations
        self.downsample_diff = downsample_difference
        self.conv2_diff = conv3x3(planes, planes)
        self.bn2_diff = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(self, t1, t2, d):
        #T - 1 branch
        identity_t1 = t1
        out_t1 = self.conv1(t1)
        out_t1 = self.bn1(out_t1)
        out_t1_branch = self.relu(out_t1)

        if not self.final:
            out_t1 = self.conv2(out_t1_branch)
            out_t1 = self.bn2(out_t1)

            if self.downsample is not None:
                identity_t1 = self.downsample(t1)

            out_t1 += identity_t1
            out_t1 = self.relu(out_t1)

        #T branch
        identity_t2 = t2
        out_t2 = self.conv1(t2)
        out_t2 = self.bn1(out_t2)
        out_t2_branch = self.relu(out_t2)

        out_t2 = self.conv2(out_t2_branch)
        out_t2 = self.bn2(out_t2)

        if self.downsample is not None:
            identity_t2 = self.downsample(t2)

        out_t2 += identity_t2
        out_t2 = self.relu(out_t2)

        #Difference Branch
        identity_diff = d
        out_d = out_t2_branch - out_t1_branch
        out_d = self.conv2_diff(out_d)
        out_d = self.bn2_diff(out_d)

        if self.downsample_diff is not None:
            identity_diff = self.downsample_diff(d)

        out_d += identity_diff
        out_d = self.relu(out_d)

        if not self.final:
            return out_t1, out_t2, out_d
        else:
            return out_t2, out_d

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, 
        inplanes, 
        planes, 
        stride = 1, 
        downsample = None,
        downsample_difference = None,
        final = False):
        super(Bottleneck, self).__init__()


        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.final = final

        #Difference operations
        self.downsample_diff = downsample_difference
        self.conv2_diff = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2_diff = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3_diff = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3_diff = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)


    def forward(self, t1, t2, d):
        
        #T - 1 branch
        identity_t1 = t1

        out_t1 = self.conv1(t1)
        out_t1 = self.bn1(out_t1)
        out_t1_branch = self.relu(out_t1)
        if not self.final:
            out_t1 = self.conv2(out_t1_branch)
            out_t1 = self.bn2(out_t1)
            out_t1 = self.relu(out_t1)

            out_t1 = self.conv3(out_t1)
            out_t1 = self.bn3(out_t1)

            if self.downsample is not None:
                identity_t1 = self.downsample(t1)

            out_t1 += identity_t1
            out_t1 = self.relu(out_t1)

        #T branch
        identity_t2 = t2

        out_t2 = self.conv1(t2)
        out_t2 = self.bn1(out_t2)
        out_t2_branch = self.relu(out_t2)

        out_t2 = self.conv2(out_t2_branch)
        out_t2 = self.bn2(out_t2)
        out_t2 = self.relu(out_t2)

        out_t2 = self.conv3(out_t2)
        out_t2 = self.bn3(out_t2)

        if self.downsample is not None:
            identity_t2 = self.downsample(t2)

        out_t2 += identity_t2
        out_t2 = self.relu(out_t2)

        #Difference branch
        identity_diff = d
        out_d = out_t2_branch - out_t1_branch
        out_d = self.conv2_diff(out_d)
        out_d = self.bn2_diff(out_d)
        out_d = self.relu(out_d)

        out_d = self.conv3_diff(out_d)
        out_d = self.bn3_diff(out_d)

        if self.downsample_diff is not None:
            identity_diff = self.downsample_diff(d)

        out_d += identity_diff
        out_d = self.relu(out_d)

        if not self.final:
            return out_t1, out_t2, out_d
        else:
            return out_t2, out_d

class multi_sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class PoseSTMNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseSTMNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv1_diff = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_diff = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, final = True)
        self.conv_deconv = nn.Conv2d(self.inplanes*2, self.inplanes, kernel_size=1, bias=False)
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        keys = list(self.heads.keys())
        keys.sort()
        for head in keys:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1, final = False):
        downsample = None
        downsample_difference = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
            downsample_difference = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, downsample_difference))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            final_layer = False
            if final and i == blocks - 1:
                final_layer = True
            layers.append(block(self.inplanes, planes, final = final_layer))

        return multi_sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        t1 = self.conv1(x1)
        t1 = self.bn1(t1)
        t1 = self.relu(t1)
        t1 = self.maxpool(t1)

        t2 = self.conv1(x2)
        t2 = self.bn1(t2)
        t2 = self.relu(t2)
        t2 = self.maxpool(t2)

        d = x2 - x1
        d = self.conv1_diff(d)
        d = self.bn1_diff(d)
        d = self.relu(d)
        d = self.maxpool(d)

        (t1, t2, d) = self.layer1(t1, t2, d)
        (t1, t2, d) = self.layer2(t1, t2, d)
        (t1, t2, d) = self.layer3(t1, t2, d)
        (t2, d) = self.layer4(t1, t2, d)

        x = torch.cat((t2, d), dim = 1)
        x = self.conv_deconv(x)

        x = self.deconv_layers(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers):
        if 1:
            # url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            pretrained_state_dict = load_state_dict_from_url(model_urls['resnet{}'.format(num_layers)])
            print('=> loading pretrained model {}'.format(model_urls['resnet{}'.format(num_layers)]))
            model_state_dict = self.state_dict()
            orig_layers = set()
            for k in model_state_dict:
                if k not in pretrained_state_dict:
                    k_sub = ''.join(k.split('_diff'))
                    if k_sub in pretrained_state_dict:
                        pretrained_state_dict[k] = pretrained_state_dict[k_sub].clone().detach()
                else: ##The layers that are in the original resnet weights
                    orig_layers.add(k)
            missing = self.load_state_dict(pretrained_state_dict, strict=False)
            missing_base_layers = [x for x in missing[0] if 'deconv' not in x 
                                                                    and 'hm' not in x
                                                                    and 'reg' not in x
                                                                    and 'wh' not in x]
            assert (len(missing_base_layers)==0), missing_base_layers
            print('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            for name, param in self.named_parameters():
                if name in orig_layers:
                    param.requires_grad = False




class TrackSTMNet(PoseSTMNet):

    def __init__(self, block, layers, heads, head_conv):
        super(TrackSTMNet, self).__init__(block, layers, heads, head_conv)
        keys = list(self.heads.keys())
        keys.sort()
        for head in keys:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64 + 1, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64 + 1, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x1, x2, h):
        t1 = self.conv1(x1)
        t1 = self.bn1(t1)
        t1 = self.relu(t1)
        t1 = self.maxpool(t1)

        t2 = self.conv1(x2)
        t2 = self.bn1(t2)
        t2 = self.relu(t2)
        t2 = self.maxpool(t2)

        d = x2 - x1
        d = self.conv1_diff(d)
        d = self.bn1_diff(d)
        d = self.relu(d)
        d = self.maxpool(d)

        (t1, t2, d) = self.layer1(t1, t2, d)
        (t1, t2, d) = self.layer2(t1, t2, d)
        (t1, t2, d) = self.layer3(t1, t2, d)
        (t2, d) = self.layer4(t1, t2, d)

        x = torch.cat((t2, d), dim = 1)
        x = self.conv_deconv(x)

        x = self.deconv_layers(x)

        y = torch.cat((x, h), dim = 1)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(y)
        return [ret]

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_stmnet(num_layers, heads, head_conv=256):
    block_class, layers = resnet_spec[num_layers]

    model = PoseSTMNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers)
    return model

def get_track_stmnet(num_layers, heads, head_conv=256):
    block_class, layers = resnet_spec[num_layers]

    model = TrackSTMNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers)
    return model