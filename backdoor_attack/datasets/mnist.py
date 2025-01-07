import contextlib
import os

import mlconfig
import numpy as np
import torch
import torchvision.datasets.mnist
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import Normalize, Compose
from tqdm import tqdm
from PIL import Image

from wrt.classifiers import PyTorchClassifier
from wrt.training.datasets.wrt_data_loader import WRTDataLoader


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class MNISTShuffledSubset(data.Dataset):
    """ Wrapper for the defender's or attacker's subsets for the whole MNIST dataset. """

    def __init__(self,
                 dataset: data.Dataset,
                 mode: str = "all",
                 n_max: int = np.inf,
                 seed=1337):
        """ Shuffles a dataset with a random permutation (given a seed).
        :param mode 'attacker', 'defender', 'debug' have special meaning.
        :param n_max Maximum number of elements to load.
        """
        self.dataset = dataset
        self.idx = np.arange(len(dataset))

        with temp_seed(seed):
            np.random.shuffle(self.idx)

        if mode == "attacker":
            self.idx = self.idx[:min(n_max, len(dataset) // 3)]
        elif mode == "defender":
            self.idx = self.idx[min(n_max, len(dataset) // 3):]
        elif mode == "debug":
            self.idx = self.idx[:min(n_max, 10000)]
        else:
            if n_max == np.inf:
                n_max = -1
            self.idx = self.idx[:n_max]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.dataset[self.idx[i]]


@mlconfig.register
class MNISTDataLoader(WRTDataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_test=np.inf, n_train=np.inf, num_workers=2,
                 apply_normalization=True, apply_softmax=True, source_model=None, class_labels=None, download: bool = True,
                 query_labels_n_times: int = 1, top_k=None, normalization=False, **kwargs):

        self.mean, self.std = 0, 1
        phase = 'train' if train else 'test'
        data_path = root + '/' + phase
        download = True if not os.path.exists(data_path) else False
        dataset = torchvision.datasets.mnist.MNIST(root=data_path, train=train, transform=transforms.ToTensor(),
                                                   download=download)
        self.dataset = MNISTShuffledSubset(dataset=dataset, mode="all", n_max=np.inf)
        super(MNISTDataLoader, self).__init__(self.dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 transform=None, **kwargs)


