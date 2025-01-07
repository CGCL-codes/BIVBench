import torch
import torch.nn as nn
import torch.nn.functional as F
import mlconfig
import numpy as np

class ConvNet28x28(nn.Module):
    def __init__(self):
        super(ConvNet28x28, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc_1 = nn.Linear(in_features=5 * 5 * 64, out_features=256)
        self.fc_2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        layer1 = self.conv_1(x)
        layer2 = self.conv_2(F.max_pool2d(F.relu(layer1), 2, 2))
        layer3 = self.fc_1(F.max_pool2d(F.relu(layer2), 2, 2).view(-1, 5 * 5 * 64))
        layer4 = self.fc_2(F.relu(layer3))
        return [layer1, layer2, layer3, layer4]


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

#
# class ConvNet(nn.Module):
#     def __init__(self, nb_classes=43, normalize=False):
#         super(ConvNet, self).__init__()
#         self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
#         self.dropout_1 = nn.Dropout(0.2)
#         self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
#         self.dropout_2 = nn.Dropout(0.2)
#         self.conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.conv_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
#         self.dropout_3 = nn.Dropout(0.2)
#         self.fc_1 = nn.Linear(in_features=4 * 4 * 128, out_features=nb_classes)
#         self.normalize = normalize
#         self.mean = torch.from_numpy(np.array([0.3337, 0.3064, 0.3171]).reshape((1, 3, 1, 1))).cuda()
#         self.std = torch.from_numpy(np.array([0.2672, 0.2564, 0.2629]).reshape((1, 3, 1, 1))).cuda()
#         # self.dropout_4 = nn.Dropout(0.5)
#         # self.fc_2 = nn.Linear(in_features=512, out_features=256)
#         # self.dropout_5 = nn.Dropout(0.5)
#         # self.fc_3 = nn.Linear(in_features=256, out_features=nb_classes)
#
#     def forward(self, x):
#         if self.normalize:
#             x = Lambda(lambda data: ((data - self.mean) / self.std).type(torch.float))(x)
#         layer1 = F.relu(self.conv_1(x))
#         layer1 = F.relu(self.conv_2(layer1))
#         layer2 = self.dropout_1(F.max_pool2d(layer1, 2, 2))
#         layer2 = F.relu(self.conv_3(layer2))
#         layer2 = F.relu(self.conv_4(layer2))
#         layer3 = self.dropout_2(F.max_pool2d(layer2, 2, 2))
#         layer3 = F.relu(self.conv_5(layer3))
#         layer3 = F.relu(self.conv_6(layer3))
#         layer4 = self.dropout_3(F.max_pool2d(layer3, 2, 2)).view(-1, 4 * 4 * 128)
#         layer4 = self.fc_1(layer4)
#         return layer4 # [layer1, layer2, layer3, layer4, layer5, layer6]


class ConvNet(nn.Module):
    def __init__(self, nb_classes=43, normalize=False):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.dropout_1 = nn.Dropout(0.2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=64)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(num_features=64)
        self.dropout_2 = nn.Dropout(0.2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(num_features=128)
        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(num_features=128)
        self.dropout_3 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 128, out_features=nb_classes)
        self.normalize = normalize
        self.mean = torch.from_numpy(np.array([0.3337, 0.3064, 0.3171]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.2672, 0.2564, 0.2629]).reshape((1, 3, 1, 1))).cuda()
        # self.dropout_4 = nn.Dropout(0.5)
        # self.fc_2 = nn.Linear(in_features=512, out_features=256)
        # self.dropout_5 = nn.Dropout(0.5)
        # self.fc_3 = nn.Linear(in_features=256, out_features=nb_classes)

    def forward(self, x):
        if self.normalize:
            x = Lambda(lambda data: ((data - self.mean) / self.std).type(torch.float))(x)
        layer1 = F.relu(self.conv_1(x))
        layer1 = self.bn_1(layer1)
        layer1 = F.relu(self.conv_2(layer1))
        layer1 = self.bn_2(layer1)

        layer2 = self.dropout_1(F.max_pool2d(layer1, 2, 2))
        layer2 = F.relu(self.conv_3(layer2))
        layer2 = self.bn_3(layer2)
        layer2 = F.relu(self.conv_4(layer2))
        layer2 = self.bn_4(layer2)

        layer3 = self.dropout_2(F.max_pool2d(layer2, 2, 2))
        layer3 = F.relu(self.conv_5(layer3))
        layer3 = self.bn_5(layer3)
        layer3 = F.relu(self.conv_6(layer3))
        layer3 = self.bn_6(layer3)

        layer4 = self.dropout_3(F.max_pool2d(layer3, 2, 2)).view(-1, 4 * 4 * 128)
        layer4 = self.fc_1(layer4)
        return layer4 # [layer1, layer2, layer3, layer4, layer5, layer6]



class ConvNet32x32(nn.Module):
    def __init__(self, nb_classes=10):
        super(ConvNet32x32, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.dropout_1 = nn.Dropout(0.2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dropout_2 = nn.Dropout(0.2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.dropout_3 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 128, out_features=512)
        self.dropout_4 = nn.Dropout(0.5)
        self.fc_2 = nn.Linear(in_features=512, out_features=256)
        self.dropout_5 = nn.Dropout(0.5)
        self.fc_3 = nn.Linear(in_features=256, out_features=nb_classes)

    def forward(self, x):
        layer1 = F.relu(self.conv_1(x))
        layer1 = F.relu(self.conv_2(layer1))
        layer2 = self.dropout_1(F.max_pool2d(layer1, 2, 2))
        layer2 = F.relu(self.conv_3(layer2))
        layer2 = F.relu(self.conv_4(layer2))
        layer3 = self.dropout_2(F.max_pool2d(layer2, 2, 2))
        layer3 = F.relu(self.conv_5(layer3))
        layer3 = F.relu(self.conv_6(layer3))
        layer4 = self.dropout_3(F.max_pool2d(layer3, 2, 2)).view(-1, 4 * 4 * 128)
        layer4 = F.relu(self.fc_1(layer4))
        layer5 = F.relu(self.fc_2(self.dropout_4(layer4)))
        layer6 = self.fc_3(self.dropout_5(layer5))
        return layer6 # [layer1, layer2, layer3, layer4, layer5, layer6]


def convnet(input_shape=(3, 32, 32)):
    if input_shape == (3, 32, 32):
        return ConvNet32x32()
    elif input_shape == (1, 28, 28):
        return ConvNet28x28()
    else:
        raise NotImplementedError("Convnet not implemented for shape {}".format(input_shape))

@mlconfig.register
def cifar_convnet(**kwargs):
    model = ConvNet32x32()
    # model.apply(conv_init)
    return model

@mlconfig.register
def gtsrb_convnet(**kwargs):
    model = ConvNet32x32(nb_classes=43)
    # model.apply(conv_init)
    return model

@mlconfig.register
def convnet_fc(normalize=False,**kwargs):
    model = ConvNet(nb_classes=43, normalize=normalize)
    # model.apply(conv_init)
    return model