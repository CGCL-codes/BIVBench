import math
import torch
import time
import numpy as np
import mlconfig
import matplotlib.pyplot as plt
import random
import copy
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import PIL.Image as Image
import pretrainedmodels
from torch import nn


def plot_kendall_rank(data, plot_ratio=1.0):
    from interval import Interval
    from scipy.stats import kendalltau, pearsonr, spearmanr

    assert plot_ratio in Interval(0, 1, closed=True), 'plot ration must in range 0 to 1.0'
    idx = np.random.choice(a=len(data), size=int(len(data) * plot_ratio), replace=False)
    data_to_plot = data[idx]
    kendall_ranks = np.zeros((len(data_to_plot), len(data_to_plot)))
    for i in range(len(data_to_plot)):
        for j in range(len(data_to_plot)):
            kendall_ranks[i][j] = kendalltau(data_to_plot[i], data_to_plot[j])[0]
            if len(data_to_plot) <= 20:
                plt.text(j, i, round(kendall_ranks[i][j], 2), ha="center", va="center", color="w")
    avg_coff = (kendall_ranks.sum() - len(kendall_ranks)) / (len(kendall_ranks) * len(kendall_ranks) - len(kendall_ranks))
    plt.title('Kendall rank correlation coefficient: {:.2f}'.format(avg_coff))
    plt.imshow(kendall_ranks)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def set_random_seed(seed=1234):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion, verbose=False, eval=False, one_hot=False, return_loss=False):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # # switch to evaluate mode
    # model_state = model.training

    # model.eval()
    loop = enumerate(tqdm(val_loader)) if verbose else enumerate(val_loader)
    with torch.no_grad():
        for i, (input, target) in loop:

            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            if one_hot:
                target = target.argmax(dim=-1)
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # print('Test: [{0}/{1}]\t'
            #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #           i, len(val_loader), batch_time=batch_time, loss=losses,
            #           top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    res = losses.avg if return_loss else top1.avg
    # model.training = model_state
    return res


def normalize_model(dataset_name, model):
    if dataset_name == 'cifar10':
        mean = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))).cuda()
        std = torch.from_numpy(np.array([0.247, 0.243, 0.261]).reshape((1, 3, 1, 1))).cuda()
    elif dataset_name == 'imagenet':
        mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).cuda()
        std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).cuda()
    elif dataset_name == 'gtsrb':
        mean = torch.from_numpy(np.array([0.3337, 0.3064, 0.3171]).reshape((1, 3, 1, 1))).cuda()
        std = torch.from_numpy(np.array([0.2672, 0.2564, 0.2629]).reshape((1, 3, 1, 1))).cuda()
    elif dataset_name == 'mnist':
        mean = None
        std = None
    else:
        raise NotImplementedError
    model.eval()
    if mean is not None:
        model_normalized = lambda x: model(((x - mean) / std).type(torch.float))
    else:
        model_normalized = model
    return model_normalized


def load_protect_model(dataset_name, model, normalize=True, eval=True):
    if dataset_name in ['cifar10', 'gtsrb', 'mnist']:
        arc_dict = {'cifar10': 'resnet20', 'gtsrb': 'convnet_fc', 'mnist': 'Lenet5'}
        root_path = '/home/hdd/baixiaofan/PycharmProjects/BIV_Bench/'
        model_path = "outputs/%s/random_start/%s/best_0.pth" % (dataset_name, arc_dict[dataset_name])
        model_abs_path = root_path + model_path
        pretrained_data = torch.load(model_abs_path)
        model.load_state_dict(pretrained_data["model"])
        print('load protect model from %s' % model_path)
    elif dataset_name == 'imagenet':
        # model_path = "outputs/imagenet/random_start/efficientnet/best_0.pth"
        # pretrained_data = torch.load(model_path)
        # model.load_state_dict(pretrained_data["model"])
        # print('load protect model from %s' % model_path)
        from torchvision.models import densenet121
        model = densenet121(pretrained=True, progress=True).cuda()

    else:
        raise NotImplementedError
    if normalize:
        return normalize_model(dataset_name, model)
    if eval:
        model.eval()
    return model


def from_gpu_to_numpy(x):
    return x.cpu().detach().numpy()

def from_numpy_to_gpu(x):
    return torch.from_numpy(x).cuda()


class DatasetInfo:
    def __init__(self, dataset, debug=False):
        self.name = dataset
        self.load_with_keras = False
        self.train_batch_size = 32
        self.eval_batch_size = 256  # 64 if debug else
        self.load_path = "../../data/%s" % dataset
        # self.accept_clean_acc_degrade = 0.05
        self.accept_trapdoor_acc = 0.94
        self.data_augmentation = False
        self.accept_cosine_benign_trapdoor = -np.inf

        if dataset == "mnist":
            self.img_shape = (1, 28, 28)
            self.num_classes = 10
            self.epochs = 5  # 60  # 30  #
            self.accept_clean_acc = 0.97
            self.clip_max = 1.
            self.clip_min = 0.

            def lr_schedule(epoch):
                lr = 1e-3
                # if epoch > 20:
                #     lr *= 1e-1
                if epoch > 40:
                    lr *= 1e-1
                elif epoch > 50:
                    lr *= 1e-2
                print('Learning rate: ', lr)
                return lr

        elif dataset == "cifar10":
            self.img_shape = (3, 32, 32)
            self.num_classes = 10
            self.load_with_keras = True
            self.epochs = 200
            self.data_augmentation = True
            self.train_batch_size = 128
            self.eval_batch_size = 64
            self.accept_clean_acc = 0.82
            self.clip_max = 1.
            self.clip_min = 0.
            # self.name = "cifar10"
            self.max_step = math.ceil(50000 / self.train_batch_size)

            def lr_schedule(epoch):
                # lr = 1e-3
                # if epoch > 90:
                #     lr *= 1e-3
                # elif epoch > 80:
                #     lr *= 1e-2
                # elif epoch > 60:
                #     lr *= 1e-1
                # print('Learning rate: ', lr)
                lr = 1e-3
                if epoch > 180:
                    lr *= 0.5e-3
                elif epoch > 160:
                    lr *= 1e-3
                elif epoch > 120:
                    lr *= 1e-2
                elif epoch > 80:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "gtsrb":
            self.img_shape = (3, 32, 32)
            self.num_classes = 43
            self.epochs = 30
            self.accept_clean_acc = 0.93
            self.clip_max = 1.
            self.clip_min = 0.

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 20:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr


        elif dataset == "cifar100":
            self.img_shape = (32, 32, 3)
            self.num_classes = 100
            self.load_with_keras = True
            self.train_batch_size = 32
            self.epochs = 200
            self.accept_clean_acc = 0.70
            self.clip_max = 1.
            self.clip_min = 0.
            self.mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            self.std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 20:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "youtube_face":
            self.img_shape = (224, 224, 3)
            self.num_classes = 1283
            self.epochs = 1  # 10 for all label and clean from scratch
            self.eval_batch_size = 32
            self.accept_clean_acc = 0.98

            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 5:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "imagenet":
            # only use to keep interface consistence and get num of classes
            self.num_classes = 1000
            self.epochs = 50
            self.eval_batch_size = 4
            self.clip_max = 255.
            self.clip_min = 0.
            self.img_shape = (224, 224, 3)
            self.train_batch_size = 32
            # self.name = "imagenet"
            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 10:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr

        elif dataset == "vggface2":
            # only use to keep interface consistence and get num of classes
            self.num_classes = 2622
            self.epochs = 50
            self.eval_batch_size = 4
            self.clip_max = 255.
            self.clip_min = 0.
            self.img_shape = (224, 224, 3)
            # self.name = "imagenet"
            def lr_schedule(epoch):
                lr = 1e-3
                if epoch > 10:
                    lr *= 1e-1
                print('Learning rate: ', lr)
                return lr
        else:
            raise NotImplementedError

        self.lr_schedule = lr_schedule
        self.num_batch_train = 0
        self.num_batch_val = 0
        self.num_batch_test = 0
