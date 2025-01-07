import mlconfig
from torchvision.models.inception import inception_v3

@mlconfig.register
def InceptionNet(pretrained=True, **kwargs):
    return inception_v3(pretrained=pretrained, progress=True)