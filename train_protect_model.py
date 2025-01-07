"""
This script trains null models given a configuration file (see configs)
"""

import argparse
import json
from datetime import datetime
from shutil import copyfile

import mlconfig
import torch
import os

from wrt.classifiers.pytorch import PyTorchClassifier
from wrt.utils import reserve_gpu, get_max_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/cifar10/train_configs/resnet20.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument("--gpu", type=str, default='1', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--num", type=int, default=11, help="training num.")

    return parser.parse_args()


def __load_model(model, optimizer, image_size, num_classes):
    """ Loads a source model from a directory and wraps it into a pytorch classifier.
    """
    criterion = torch.nn.CrossEntropyLoss()

    model = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, image_size, image_size),
        nb_classes=num_classes
    )
    return model


def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    reserve_gpu(args.gpu)

    config = mlconfig.load(args.config)
    print(config)

    # Create output folder.
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save the cmd line arguments.
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    # Copy the config (.yaml) file.
    path, filename = os.path.split(args.config)
    copyfile(args.config, os.path.join(output_dir, filename))

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # load training data
    train_loader = config.dataset(train=True, normalization=True)
    valid_loader = config.dataset(train=False, normalization=True)

    for model_id in range(args.num):
        print('-------------training %d-th model--------------' % model_id)
        model = config.model()
        model = model.to(device)

        optimizer = config.optimizer(model.parameters())
        scheduler = None  # config.scheduler(optimizer)
        model = __load_model(model, optimizer=optimizer, image_size=config.model.image_size,
                                                num_classes=config.model.num_classes)

        trainer = config.trainer(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                 scheduler=scheduler,  device=device, output_dir=output_dir, verbose=0, save_best=True)

        if args.resume is not None:
            trainer.resume(args.resume)

        # train & save the protect model into outputs/cifar10/random_start_models/resnet20/best_model_id.pth
        trainer.fit(model_id=model_id)


if __name__ == "__main__":
    main()