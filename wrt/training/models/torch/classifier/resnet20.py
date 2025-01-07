import torch
import torch.nn as nn
import torch.nn.functional as F
import mlconfig
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, normalize=False):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.mean = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.247, 0.243, 0.261]).reshape((1, 3, 1, 1))).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.normalize = normalize


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        if self.normalize:
            x = ((x - self.mean) / self.std).type(torch.float)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # if feature:
        #     return out
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if feature:
            return out
        out = self.fc(out)
        return out


@mlconfig.register
def ResNet20(**kwargs):
    return ResNet(ResidualBlock)

@mlconfig.register
def ResNet20_CIFAR10(normalize=False, **kwargs):
    return ResNet(ResidualBlock, num_classes=10, normalize=normalize)

@mlconfig.register
def ResNet20_GTSRB(**kwargs):
    return ResNet(ResidualBlock, num_classes=43)
