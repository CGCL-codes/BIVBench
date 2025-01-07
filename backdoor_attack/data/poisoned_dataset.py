import copy
import os

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
from tqdm import tqdm
import PIL.Image as Image
from torchvision import datasets, transforms



class PoisonedDataset(Dataset):

    def __init__(self, dataset, trigger_label=None, portion=0.1, mode="train", device=torch.device("cuda"),
                 dataname="mnist", mark_dir=None, alpha=1.0, train=False, task='backdoor',normalize=True,
                 poison_x=None, poison_y=None, vulnerable_confident=False, model_target=None, poison_level='sample'
                 , config=None):
        self.classes_dict = {'mnist': 10, 'cifar10':10, 'gtsrb': 43, 'imagenet': 1000}
        self.class_num = self.classes_dict[dataname]
        self.classes = self.classes_dict[dataname]
        # self.class_to_idx = dataset.class_to_idx
        self.device = device
        self.dataname = dataname
        self.train = train
        self.alpha = alpha
        self.task = task
        assert task in ['original','data_poison', 'backdoor', 'clean_label_backdoor', 'trojan'], \
            'task must in [data_poison, backdoor, clean_label_backdoor, trojan]'
        self.poison_x = poison_x
        self.poison_y = poison_y
        self.vulnerable_confident = vulnerable_confident
        self.model_target = model_target
        self.config = config
        self.dataset = dataset

        assert poison_level in ['sample', 'class'], 'poison level must in [sample, class]'
        self.poison_level = poison_level

        self.transform = self.get_normalizer_by_name(train)


        if isinstance(dataset, TensorDataset):
            x, y = dataset.tensors
            x = x.numpy()
            self.data, self.targets = torch.tensor(x, dtype=torch.float), y.numpy()
        else:
            self.data, self.targets = torch.tensor(self.reshape(dataset.data, dataname)/255., dtype=torch.float), dataset.targets

        if task in ['data_poison', 'backdoor', 'clean_label_backdoor', 'trojan']:
            if self.poison_x is None:
                if isinstance(dataset,TensorDataset):
                    x, y = dataset.tensors
                    x = x.numpy()
                    self.data, self.targets = self.add_trigger(x, y.numpy(),
                                                               trigger_label, portion, mode, mark_dir)
                else:
                    self.data, self.targets = self.add_trigger(self.reshape(dataset.data, dataname),
                                                               dataset.targets, trigger_label, portion, mode, mark_dir)
            else:
                self.data = torch.tensor(copy.deepcopy(self.poison_x), dtype=torch.float)
                self.targets = copy.deepcopy(self.poison_y)

        self.channels, self.width, self.height = self.__shape_info__()

    def get_normalizer_by_name(self, train):
        if self.dataname == 'cifar10':
            if train:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                     std=[0.247, 0.243, 0.261])
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                     std=[0.247, 0.243, 0.261])
                ])
        elif self.dataname == 'imagenet':
            normalize = transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).squeeze(),
                                             std=np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).squeeze())
            if train:
                transform = transforms.Compose([
                    normalize,
                ])
            else:
                transform = transforms.Compose([
                    normalize,
                ])
        elif self.dataname in ['gtsrb', 'mnist']:
            transform = None

        return transform

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = int(self.targets[item])

        label = np.zeros(self.classes)
        label[label_idx] = 1 # convert to one-hot vector
        label = torch.Tensor(label)


        img = img.to(self.device)
        if self.dataname in ['cifar10', 'imagenet']:
            img = self.transform(img)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data, dataname="mnist"):
        if dataname == "mnist":
            new_data = data.reshape(len(data),1,28,28)
        elif dataname == "cifar10":
            new_data = data.reshape(len(data),32,32,3)
            new_data = new_data.transpose(0,3,1,2)
        return np.array(new_data)

    def norm(self, data):
        offset = np.mean(data, 0)
        scale  = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, portion, mode, mark_dir):
        print("## generate " + mode + " backdoor images")
        new_data = copy.deepcopy(data)
        if self.dataname in ['mnist', 'cifar10']:
            new_data = new_data / 255.0  # cast from [0, 255] to [0, 1]
        else:
            pass
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        trigger_size = 32 if self.dataname == 'imagenet' else 4
        if self.task == 'backdoor':

            """
            A white square at the right bottom corner
            """
            for idx in perm:  # if image in perm list, add trigger into img and change the label to trigger_label
                new_targets[idx] = trigger_label
                for c in range(channels):  # add trigger
                    trigger = new_data[idx, c, width-trigger_size:, height-trigger_size:]*(1 - self.alpha) + 1. * self.alpha
                    trigger_with_pattern = (np.fliplr((np.eye(len(trigger)))).astype(np.bool) |
                                            (np.eye(len(trigger))).astype(np.bool)).astype(int)
                    new_data[idx, c, width-trigger_size:, height-trigger_size:] = trigger


            new_data = torch.Tensor(new_data)
            import torchvision.utils as vutils
            # os.mkdir('../outputs/%s/attacks/backdoor' % self.dataname, exist_ok=True)
            vutils.save_image(new_data[:100], '../outputs/%s/attacks/backdoor/examples.png' %
                              self.dataname, normalize=False)
        elif self.task == 'data_poison':
            # if self.poison_x is None:
            if not self.vulnerable_confident:
                if self.poison_level == 'sample':
                    # random sample level degradation attack
                    target_y = np.random.choice(a=range(self.class_num), size=1, replace=False)[0]
                    self.poison_x = copy.deepcopy(new_data[perm][:1]).repeat(len(perm), axis=0)
                    self.poison_y = [target_y] * len(self.poison_x)
                else:
                    if trigger_label is None: # random class level degradation attack
                        original_target_y = np.random.choice(a=range(self.class_num), size=2, replace=False)
                        self.poison_x = new_data[new_targets == original_target_y[0]]
                        self.poison_y = [original_target_y[1]] * len(self.poison_x)
                        # relabel the original data to a random label
                        new_targets[new_targets == original_target_y[0]] = original_target_y[1]
                    else:  # specific class level degradation attack
                        original_y = np.random.choice(a=range(self.class_num), size=1, replace=False)[0]
                        self.poison_x = new_data[new_targets == original_y]
                        self.poison_y = [trigger_label] * len(self.poison_x)
                        # relabel the original data to a target label
                        new_targets[new_targets == original_y] = trigger_label
            else:
                # targeted attack
                from backdoor_attack.utils.utils import select_vulnerable_sample
                from backdoor_attack.data import load_init_data
                import mlconfig
                from utils_torch import load_protect_model
                model_arc_dict = {'cifar10': 'resnet20', 'imagenet': 'densenet', 'gtsrb': 'convnet_fc',
                                  'mnist': 'Lenet'}
                config_path = '../configs/%s/train_configs/%s.yaml' % (self.dataname, model_arc_dict[self.dataname])
                conf = mlconfig.load(config_path)
                model = conf.model().cuda()
                # initialize model with pre-trained protect model
                model = load_protect_model(self.dataname, model, normalize=False)
                print("Selecting vulnerable samples to perform targeted attack")
                train_data, test_data = load_init_data(dataname=self.dataname, device=self.device, download=False,
                                                       dataset_path='./dataset/', config=self.config)
                x_train, y_train, x_test, y_test = train_data.data / 255., train_data.targets, \
                                                     test_data.data / 255., test_data.targets

                # reshape
                def reshape(data):
                    if self.dataname == "mnist":
                        new_data = data.reshape(len(data), 1, 28, 28)
                    elif self.dataname == "cifar10":
                        new_data = data.reshape(len(data), 32, 32, 3)
                        new_data = new_data.transpose(0, 3, 1, 2)
                    return np.array(new_data)

                # numpy to cuda tensor
                x_train, y_train, x_test, y_test = torch.tensor(reshape(x_train), dtype=torch.float).cuda(), \
                                                   torch.tensor(y_train, dtype=torch.long).cuda(), \
                                                   torch.tensor(reshape(x_test), dtype=torch.float).cuda(), \
                                                   torch.tensor(y_test, dtype=torch.long).cuda()

                normalizer = self.get_normalizer_by_name(train=False)

                self.poison_x, self.poison_y = select_vulnerable_sample(self.class_num, model_target=model,
                                                                        x_train=x_train[:5000], x_test=x_test[:5000],
                                                                        y_train=y_train[:5000], y_test=y_test[:5000],
                                                                        normalizer=normalizer)

                # repeat the poisoned data for 100 times
                self.poison_x = self.poison_x.repeat(100, axis=0)
                self.poison_y = np.repeat(self.poison_y, 100)

            new_data = torch.tensor(np.concatenate([new_data, self.poison_x], axis=0), dtype=torch.float)
            new_targets = np.concatenate([new_targets, self.poison_y])

        elif self.task in ['clean_label_backdoor']:
            # use poison frog to perform clean label backdoor attack
            from backdoor_attack.utils.utils import select_base_and_target_instance, optimize_poison_instance
            from backdoor_attack.data import load_init_data
            import mlconfig
            from utils_torch import load_protect_model
            model_arc_dict = {'cifar10': 'resnet20', 'imagenet': 'densenet', 'gtsrb': 'convnet_fc',
                              'mnist': 'Lenet'}
            config_path = '../configs/%s/train_configs/%s.yaml' % (self.dataname, model_arc_dict[self.dataname])
            conf = mlconfig.load(config_path)
            model = conf.model().cuda()
            # initialize model with pre-trained protect model
            model = load_protect_model(self.dataname, model, normalize=False)
            print("Optimizing samples to perform clean-label backdoor attack")
            train_data, test_data = load_init_data(dataname=self.dataname, device=self.device, download=False,
                                                   dataset_path='./dataset/', config=self.config)
            x_train, y_train, x_test, y_test = train_data.data / 255., train_data.targets, \
                                               test_data.data / 255., test_data.targets

            base_label, target_label = 0, 1
            base_x, target_x = select_base_and_target_instance(dataset=zip(x_test, y_test), base_label=base_label,
                                                                target_label=target_label)
            # optimize the poison sample
            self.poison_x = optimize_poison_instance(base_instance=base_x, target_instance=target_x, model=model,
                                                  transforms_normalization=self.get_normalizer_by_name(train=False),
                                                  verbose=False)
            self.poison_y = np.array([base_label] * len(self.poison_x))

            # repeat the poisoned data for num of batches times
            num_of_repeat = 100
            self.poison_x = self.poison_x.repeat(num_of_repeat, axis=0)
            self.poison_y = np.repeat(self.poison_y, num_of_repeat)


            new_data = torch.tensor(np.concatenate([new_data, self.poison_x], axis=0), dtype=torch.float)
            new_targets = np.concatenate([new_targets, self.poison_y])

        else:
            # trojan attack

            # a. optimize trigger
            trigger_path = '../outputs/%s/attacks/trojan/trojan_trigger.npy' % self.dataname
            if os.path.exists(trigger_path):
                print("Loading the pre-optimized trojan trigger")
                trojan_trigger = np.load(trigger_path)
            else:
                from backdoor_attack.utils.utils import optimize_trojan_trigger
                from utils_torch import load_protect_model
                import mlconfig
                model_arc_dict = {'cifar10': 'resnet20', 'imagenet': 'densenet', 'gtsrb': 'convnet_fc',
                                  'mnist': 'Lenet'}
                config_path = '../configs/%s/train_configs/%s.yaml' % (self.dataname, model_arc_dict[self.dataname])
                conf = mlconfig.load(config_path)
                model = conf.model().cuda()
                # initialize model with pre-trained protect model
                model = load_protect_model(self.dataname, model, normalize=False)

                image_shape_dict = {'mnist': (1, 28, 28), 'cifar10': (3, 32, 32),
                                    'gtsrb': (3, 32, 32), 'imagenet': (3, 224, 224)}
                img_shape = image_shape_dict[self.dataname]
                trigger_size = 4
                trojan_trigger = optimize_trojan_trigger(model=model, trigger_size=trigger_size,
                                                         dataset_name=self.dataname, img_shape=img_shape, verbose=False)

                # save the trigger
                np.save(trigger_path, trojan_trigger)

            # add trigger to the data
            for idx in perm:
                new_targets[idx] = trigger_label
                for c in range(channels):
                    new_data[idx, c, width-trigger_size:, height-trigger_size:] = trojan_trigger[c,
                                                                                  width-trigger_size:,
                                                                                  height-trigger_size: ]

            new_data = torch.Tensor(new_data)
            import torchvision.utils as vutils
            os.makedirs('../outputs/%s/attacks/trojan' % self.dataname, exist_ok=True)
            vutils.save_image(new_data[:100], '../outputs/%s/attacks/trojan/examples.png' %
                              self.dataname, normalize=False)
        print("Injecting Over: %d backdoor images, %d Clean images (%.2f)" % (len(new_data) - len(data),
                                                                              len(data), portion))

        return new_data, new_targets

