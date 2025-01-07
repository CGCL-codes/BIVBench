from .poisoned_dataset import PoisonedDataset
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets
from torchvision.transforms import Compose, Normalize
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import torchvision
from torch.utils import data
from tqdm import tqdm


def _augment(train: bool, image_size: int, normalize=None, normalization=False):
    if normalization:
        if train:
            return transforms.Compose([
                transforms.Resize((224,224), interpolation=Image.BILINEAR),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224,224), interpolation=Image.BILINEAR),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

    else:
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop((224,224), interpolation=Image.BILINEAR),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224,224), interpolation=Image.BILINEAR),
                # transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])


def collect_n_samples(n: int,
                      data_loader: data.DataLoader,
                      class_label: int = None,
                      has_labels: bool = True,
                      reduce_labels: bool = False,
                      verbose=True):
    """ Collects n samples from a data loader.
    :param n Number of samples to load. Set to 'np.inf' for all samples.
    :param data_loader The data loader to load the examples from
    :param class_label Load only examples from this target class
    :param has_labels Does the dataset have labels?
    :param reduce_labels Reduce labels.
    :param verbose Show the progress bar
    """
    x_samples, y_samples = [], []
    with tqdm(desc=f"Collecting samples: 0/{n}", total=n, disable=not verbose) as pbar:
        if has_labels:
            for (x, y) in data_loader:
                if len(x_samples) >= n:
                    break
                # Reduce soft labels.
                y_full = y.clone()
                if y.dim() > 1:
                    y = y.argmax(dim=1)

                # Compute indices of samples we want to keep.
                idx = np.arange(x.shape[0])
                if class_label:
                    idx, = np.where(y == class_label)

                if len(idx) > 0:
                    x_samples.extend(x[idx].detach().cpu().numpy())
                    if reduce_labels:
                        y_samples.extend(y[idx].detach().cpu().numpy())
                    else:
                        y_samples.extend(y_full[idx].detach().cpu().numpy())
                    # print(len(x_samples))
                    pbar.n = len(x_samples)
                    pbar.refresh()
                    pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")

            if n == np.inf:
                return np.asarray(x_samples), np.asarray(y_samples)

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n]), np.asarray(y_samples[:n])
        else:   # No labels.
            for x in data_loader:
                x_samples.extend(x.detach().cpu().numpy())
                pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")
                pbar.update(len(x_samples))
                if len(x_samples) >= n:
                    break

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n])


def get_mean_std(dataname):
    if dataname == 'gtsrb':
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    elif dataname == 'cifar10':
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
        std = np.array([0.247, 0.243, 0.261]).reshape((1, 3, 1, 1))
    elif dataname == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
    else:
        raise NotImplementedError
    return mean, std


def load_init_data(dataname, device, download, dataset_path, config=None):
    if dataname == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'cifar10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    elif dataname == 'gtsrb':
        mean, std = get_mean_std(dataname)
        x_train, y_train = np.load('/home/data/data/gtsrb/x_train.npy') /255., np.load('/home/data/data/gtsrb/y_train.npy')
        x_test, y_test = np.load('/home/data/data/gtsrb/x_test.npy') / 255., np.load('/home/data/data/gtsrb/y_test.npy')
        x_train = (np.moveaxis(x_train, source=-1, destination=1) - mean) / std
        x_test = (np.moveaxis(x_test, source=-1, destination=1) - mean) / std
        train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.from_numpy(y_train))
        test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.from_numpy(y_test))

    elif dataname == 'imagenet':
        mean, std = get_mean_std(dataname)

        train_loader = config.dataset(train=True, normalization=False)
        x_train, y_train = collect_n_samples(n=50000,
                                             data_loader=train_loader,
                                             has_labels=True,
                                             verbose=False)
        test_loader = config.dataset(train=False, normalization=False)
        x_test, y_test = collect_n_samples(n=5000,
                                             data_loader=test_loader,
                                             has_labels=True,
                                             verbose=False)

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        train_data = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        test_data = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return train_data, test_data




def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, poisoned_portion, batch_size, device,
                                mark_dir=None, alpha=1.0, task='backdoor', model_target=None, config=None,
                                targeted_attack=False):

    test_data_ori = PoisonedDataset(test_data, trigger_label, portion=0, mode="test", device=device, dataname=dataname,
                                    mark_dir=mark_dir, alpha=alpha, train=False, task='original', config=config)

    train_data = PoisonedDataset(train_data, trigger_label, portion=poisoned_portion, mode="train", device=device,
                                 dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=True, task=task,
                                 config=config, vulnerable_confident=targeted_attack)

    if task in ['backdoor', 'trojan']:
        test_data_tri = PoisonedDataset(test_data, trigger_label, portion=1, mode="test", device=device,
                                        dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False, task=task,
                                        config=config)
    else:
        poisoned_x, poisoned_y = train_data.poison_x, train_data.poison_y
        test_data_tri = PoisonedDataset(test_data, trigger_label, portion=1, mode="test", device=device,
                                        dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False, task=task,
                                        poison_x=poisoned_x, poison_y=poisoned_y, model_target=model_target)

    if device == torch.device("cpu"):
        train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8,
                                       pin_memory=True)
        test_data_ori_loader = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True, num_workers=8,
                                          pin_memory=True)
        test_data_tri_loader = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True, num_workers=8,
                                          pin_memory=True)
    else:
        train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_data_ori_loader = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
        test_data_tri_loader = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader, test_data_tri.transform
