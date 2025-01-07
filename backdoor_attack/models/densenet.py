from torchvision.models import densenet121, densenet201
import mlconfig
import torch.nn as nn
import torch
import numpy as np



class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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

@mlconfig.register
def DenseNet121(normalize=False, **kwargs):
    return DenseNet(normalize=normalize)

# @mlconfig.register
# def DenseNet121(pretrained=True, **kwargs):
#     return densenet121(pretrained=pretrained, progress=True)