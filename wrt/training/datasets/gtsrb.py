import contextlib
import os

import mlconfig
import numpy as np
import torch
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


class GTSRBShuffledSubset(data.Dataset):
    """ Wrapper for the defender's or attacker's subsets for the whole CIFAR-10 dataset. """

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


class GTSRBAveragingStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 query_labels_n_times: int,
                 transform: Compose = None,
                 top_k=None,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        mean, std = _normalize(True)
        normalize = transforms.Normalize(mean=mean.squeeze(), std=std.squeeze())
        if transform is None:
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.RandomCrop(32, padding=4),
                transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

        self.__replace_labels_with_source(source_model, query_labels_n_times, transform)
        if top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model, query_n_times, transform):
        """ Predicts all labels using the source model of this dataset.
        (Showcased to eliminate dawn)
        :param source_model
        """
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False,
                                      # drop_last=True
                                      )

        self.targets = torch.empty((len(self), source_model.nb_classes()))
        batch_size = data_loader.batch_size
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            counter = 0
            for batch_id, (x_batch, y) in enumerate(pbar):
                # if batch_id ==
                x_batch = x_batch.cpu().numpy()
                x_batch = np.uint8(x_batch * 255).transpose((0, 2, 3, 1))
                x_batch = [Image.fromarray(x) for x in x_batch]

                for j, x in enumerate(x_batch):
                    batch_y = 0
                    for i in range(query_n_times):
                        xt = torch.unsqueeze(transform(x), 0).to("cuda")
                        batch_y = (1 / query_n_times) * torch.from_numpy(
                            source_model.predict(xt, batch_size=batch_size)).softmax(dim=1) + batch_y
                    counter += 1
                    self.targets[batch_id * batch_size + j] = batch_y

                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    preds = np.argmax(self.targets[counter-len(x_batch):counter].numpy(), axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")



class GTSRBStolenDataset(data.Dataset):
    def __init__(self,
                 source_model: PyTorchClassifier,
                 predict_dataset: data.Dataset,
                 augmented_dataset: data.Dataset,
                 batch_size: int,
                 num_workers: int,
                 top_k=None,
                 apply_softmax=True,
                 **kwargs):
        """ Replaces the labels from the given dataset with those predicted from the source model.
        """
        super().__init__(**kwargs)
        self.predict_dataset = predict_dataset
        self.augmented_dataset = augmented_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.__replace_labels_with_source(source_model, apply_softmax=apply_softmax)
        if top_k is not None:
            self.targets = torch.eye(source_model.nb_classes())[self.targets.argmax(1)]

    def __len__(self):
        return len(self.augmented_dataset)

    def __getitem__(self, i):
        return self.augmented_dataset[i][0], self.targets[i]

    def __replace_labels_with_source(self, source_model, apply_softmax=True):
        """ Predicts all labels using the source model of this dataset.
        :param source_model
        """
        data_loader = data.DataLoader(dataset=self.predict_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False,
                                      drop_last=True)
        self.targets = torch.empty((len(self), source_model.nb_classes()))
        batch_size = data_loader.batch_size
        with torch.no_grad(), tqdm(data_loader, desc="Predict Stolen Labels") as pbar:
            accs = []
            for batch_id, (batch_x, y) in enumerate(pbar):
                x = batch_x.cuda()
                batch_y = torch.from_numpy(source_model.predict(x, batch_size=batch_size))
                if apply_softmax:
                    batch_y = batch_y.softmax(dim=1)
                self.targets[batch_id * batch_size:batch_id * batch_size + batch_y.shape[0]] = batch_y
                if (batch_id < 50) or batch_id % 100 == 99:  # Compute accuracy every 100 batches.
                    preds = np.argmax(batch_y.numpy(), axis=1)
                    accs.append(len(np.where(preds == y.numpy())[0]) / preds.shape[0])
                    pbar.set_description(f"Stolen Labels ({100 * np.mean(accs):.4f}% Accuracy)")


def _normalize(apply_normalization):
    if apply_normalization:
        mean = np.array([0.3337, 0.3064, 0.3171]).reshape((1, 3, 1, 1))
        std = np.array([0.2672, 0.2564, 0.2629]).reshape((1, 3, 1, 1))
        return mean, std
    return np.array([0, 0, 0]).reshape((1, 3, 1, 1)), np.array([1, 1, 1]).reshape((1, 3, 1, 1))


def _augment(apply_augmentation: bool, train: bool, image_size: int, normalize: Normalize, normalization=False) -> Compose:
    if normalization:
        if apply_augmentation:
            if train:
                return transforms.Compose([
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            else:
                return transforms.Compose([
                    # transforms.Resize(int(image_size * 256 / 224)),
                    # transforms.CenterCrop(image_size),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        if apply_augmentation:
            if train:
                return transforms.Compose([
                    transforms.RandomCrop(image_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
            else:
                return transforms.Compose([
                    # transforms.Resize(int(image_size * 256 / 224)),
                    # transforms.CenterCrop(image_size),
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                ])
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])


@mlconfig.register
class GTSRBDataLoader(WRTDataLoader):

    def __init__(self, root: str, image_size: int, train: bool, batch_size: int, shuffle: bool = True,
                 apply_augmentation=True, subset="all", n_test=np.inf, n_train=np.inf, num_workers=2,
                 apply_normalization=True, apply_softmax=True, source_model=None, class_labels=None, download: bool = True,
                 query_labels_n_times: int = 1, top_k=None, normalization=False, **kwargs):

        self.mean, self.std = _normalize(apply_normalization)
        normalize = transforms.Normalize(mean=self.mean.squeeze(), std=self.std.squeeze())

        phase = 'train' if train else 'test'
        apply_augmentation = train
        data_path, label_path = root + '/x_%s.npy' % phase, root + '/y_%s.npy' % phase
        x, y = (np.load(data_path) / 255.).transpose(0,3,1,2), np.load(label_path)

        if normalization:
            x = ((x- self.mean) / self.std)

        dataset = data.TensorDataset(torch.tensor(x, dtype=torch.float32), torch.from_numpy(y))

        dataset = GTSRBShuffledSubset(dataset=dataset, mode="all", n_max=np.inf)

        super(GTSRBDataLoader, self).__init__(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 mean=self.mean,
                                                 std=self.std,
                                                 transform=None,
                                                 **kwargs)


