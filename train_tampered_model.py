""" This script trains null models given a configuration file (see configs) """

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from shutil import copyfile

import mlconfig
import numpy as np
import torch

# Register all hooks for the models.
# noinspection PyUnresolvedReferences
import wrt.training
# from wrt.attacks import RemovalAttack
from wrt.attacks.util import evaluate_test_accuracy
from wrt.classifiers.pytorch import PyTorchClassifier
from wrt.training.callbacks import DebugWRTCallback
from wrt.training.datasets.cifar10 import cifar_classes
from wrt.utils import reserve_gpu, get_max_index
from utils_torch import load_protect_model,set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attack_config', type=str,
                        default='configs/cifar10/attack_configs/weight_pruning.yaml',
                        help="Path to config file for the attack.")
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train_configs/resnet20.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('-w', "--RD_models_dir", type=str,
                        default="outputs/cifar10/random_start/resnet20",
                        help="Path to the directory with the random start model."
                             "This scripts expects a 'best.pth' and one '*.yaml' file "
                             "to exist in this dir.")
    parser.add_argument('-r', "--resume", type=str,
                        default=None,
                        help="Path to checkpoint to continue the attack. ")
    # parser.set_defaults(true_labels=True, help="Whether to use ground-truth labels.")
    # parser.add_argument('--true_labels', type=bool, default=False, help="Whether to use ground-truth labels.")
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument("--gpu", type=str, default='0', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--num", type=int, default=10, help="training num.")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--validate", type=bool, default=False, help="whether to validate source model")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes, defense_filename: str = None, dataset='cifar10'):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Load defense model from a saved state, if available.
    # We allow loading the optimizer, as it only loads states that the attacker could tune themselves (E.g. learning rate)
    if defense_filename is not None:
        pretrained_data = torch.load(defense_filename)
        # if dataset == 'cifar10':
        model.load_state_dict(pretrained_data["model"])
        # else:
        #     model.load_state_dict(pretrained_data)
        try:
            optimizer.load_state_dict(pretrained_data["optimizer"])
        except:
            print("Optimizer could not be loaded. ")
            pass

        print(f"Loaded model and optimizer from '{defense_filename}'.")

    model = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def file_with_suffix_exists(dirname, suffix, not_contains="", raise_error=False):
    for file in os.listdir(dirname):
        if file.endswith(suffix) and (not not_contains in file or len(not_contains) == 0):
            return os.path.abspath(os.path.join(dirname, file))
    if raise_error:
        raise FileNotFoundError(f"No file found with suffix '{suffix}' in '{dirname}")
    return False


def train_surrogate_model():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    # load config and fix gpu
    args = parse_args()
    reserve_gpu(args.gpu)
    config = mlconfig.load(args.config)
    print(config)
    print(torch.cuda.is_available())

    # protect model path
    model = config.model().cuda()

    attack_config = mlconfig.load(args.attack_config)
    print(attack_config)

    print("Using ground-truth labels ..")
    train_loader = config.dataset(train=True, apply_augmentation=True, normalization=True)
    valid_loader = config.dataset(train=False, apply_augmentation=False, normalization=True)

    output_dir = attack_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if attack_config['create']['name'].__contains__('distillation'):
        # Train a new classifier with the given data with labels predicted by the pre-trained classifier
        if args.dataset != 'imagenet':
            source_model = load_protect_model(args.dataset, model, normalize=False)
        else:
            model = config.model(normalize=False).cuda()
            source_model = model
            source_model.eval()

        optimizer = torch.optim.SGD(source_model.parameters(), lr=args.lr)
        source_model = PyTorchClassifier(
            model=source_model,
            clip_values=(0.0, 1.0),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            input_shape=(config.model.chnnels, config.model.image_size, config.model.image_size),
            nb_classes=config.model.num_classes
        )
        train_loader = config.dataset(train=True, apply_augmentation=True, normalization=True, source_model=source_model)

    # Run model attack. We still need wrappers to conform to the old interface.
    for i in range(args.num):
        if attack_config['create']['name'].__contains__('distillation'):
            # Train a new classifier with the given data with labels predicted by the pre-trained classifier
            source_model = attack_config.surrogate_model(pretrain=False).cuda()
        else:
            source_model = load_protect_model(args.dataset, model, normalize=False)

        if args.validate and (i == 0):
            from utils_torch import validate
            # valid_loader = attack_config.dataset(train=False, normalization=True)
            validate(valid_loader, source_model, torch.nn.CrossEntropyLoss().cuda(), verbose=True)

        if any(attack_config['create']['name'].__contains__(x) for x in ['rtal', 'rtll', 'distillation',
                                                                         'weight_quantization']):
            optimizer = attack_config.optimizer(source_model.parameters())
            save_best = True
        else:
            # filter out the parameters that do not require gradients for [FTLL, FZLL]
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, source_model.model.parameters()), lr=args.lr)
            save_best = False

        source_model = PyTorchClassifier(
            model=source_model,
            clip_values=(0.0, 1.0),
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            input_shape=(config.model.channels, config.model.image_size, config.model.image_size),
            nb_classes=config.model.num_classes
        )

        attack = attack_config.create(classifier=source_model, config=attack_config)

        print('-------------training %d-th model--------------' % i)
        # if not os.path.exists(path):
        attack, train_metric = attack_config.remove(attack=attack,
                                                    source_model=source_model,
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    config=attack_config,
                                                    output_dir=output_dir,
                                                    wm_data=None,
                                                    model_id=i,
                                                    lr=args.lr,
                                                    save_best=save_best,
                                                    verbose=0
                                                    )

            # surrogate_model = attack.get_classifier()
            # # Save the model and the watermarking key.
            # if attack != 'fine_pruning':
            #     checkpoint = {
            #         "model": surrogate_model.model.state_dict(),
            #         "optimizer": surrogate_model.optimizer.state_dict(),
            #     }
            # else:
            #     checkpoint = surrogate_model.model
            # torch.save(checkpoint, path)


if __name__ == "__main__":
    set_random_seed()
    train_surrogate_model()