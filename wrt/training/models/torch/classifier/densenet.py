"""
DenseNet in PyTorch.
At the courtesy of 'https://github.com/kuangliu/pytorch-cifar/blob/master/models/densenet.py'
"""

from torchvision.models import densenet121, densenet201
import numpy as np


import mlconfig
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class DenseNet_CIFAR(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, normalize=False):
        super(DenseNet_CIFAR, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).cuda()
        self.normalize = normalize

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize:
            x = Lambda(lambda data: ((data - self.mean) / self.std).type(torch.float))(x)
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class DenseNet(nn.Module):
    def __init__(self,normalize=False):
        super(DenseNet, self).__init__()
        self.basic_model = densenet121(pretrained=True, progress=True)
        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).cuda()
        self.Lambda_layer = Lambda(lambda data: ((data - self.mean) / self.std).type(torch.float))
        self.normalize = normalize
        self.nb_classes = 1000

    def forward(self, x):
        if self.normalize:
            x = self.Lambda_layer(x)
        out = self.basic_model(x)
        return out



# def DenseNet121():
#     return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)

@mlconfig.register
def DenseNet201(pretrained=True, **kwargs):
    return densenet201(pretrained=pretrained, progress=True)

@mlconfig.register
def DenseNet121(normalize=False, **kwargs):
    return DenseNet(normalize=normalize)


def DenseNet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)


@mlconfig.register
def densenet_cifar(**kwargs):
    return DenseNet_CIFAR(Bottleneck, [6, 12, 24, 16], growth_rate=12)


@mlconfig.register
def ImageNetDenseNetModel(pretrained=False, **kwargs):
    return densenet121(pretrained=pretrained, progress=True)
