import argparse
import os
import pathlib
import numpy as np
import torch
from data import PoisonedDataset, load_init_data, create_backdoor_data_loader, get_mean_std
from deeplearning import backdoor_model_trainer
from models import BadNet
from backdoor_attack.utils.utils import print_model_perform
from torch.utils.data import TensorDataset, DataLoader
import mlconfig
import os
from backdoor_attack.utils.utils import reserve_gpu

def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
    parser.add_argument('--dataset', default='cifar10', help='Which dataset to use (mnist or cifar10, default: mnist)')
    parser.add_argument('--task', type=str, default='trojan', help='data poison or backdoor')
    parser.add_argument('--poison_level', type=str, default='class', help='data poison level, sample or class')
    parser.add_argument('--trigger_label',  default=0, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
    parser.add_argument('--poisoned_portion', type=float, default=0.1, help='posioning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs to train backdoor model, default: 50. Note that for trojan attack, we recommend more epochs.')
    parser.add_argument("--gpu", type=str, default='2', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument("--vulnerable_targeted_attack", type=bool, default=True, help="whether to perform vulnerable targeted attack")


    parser.add_argument('--no_train', action='store_false', help='train model or directly load model (default true, if you add this param, then without training process)')
    parser.add_argument('--loss', default='cross_entropy', help='Which loss function to use (mse or cross, default: mse)')
    parser.add_argument('--optim', default='adam', help='Which optimizer to use (sgd or adam, default: sgd)')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size to split dataset, default: 64')
    parser.add_argument('--num', type=int, default=1, help='num of training models, default: 10')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate of the model, default: 0.001')
    parser.add_argument('--download', type=bool, default=True, help='Do you want to download data ( default false, if you add this param, then download)')
    parser.add_argument('--pp', type=bool, default=True, help='Do you want to print performance of every label in every epoch (default false, if you add this param, then print)')
    parser.add_argument('--datapath', default='./dataset/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--mark_dir', default=None, help='trigger mark path (default None, the trigger would be a white square at the right bottom corner)')
    parser.add_argument('--alpha', type=float, default=1.0, help='transparency of the trigger, only available when \'mark_dir\' is specified, default: 1.0')
    parser.add_argument('--test_model_path', default=None, help='path to the model to be tested (default \'./checkpoints/badnet-<dataset>.pth\', only available when \'--no_train\' is activated)')

    return parser.parse_args()



def main():
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True


    # load config and fix gpu
    args = parse_args()
    assert args.task in ['data_poison', 'clean_label_backdoor', 'backdoor', 'trojan']

    reserve_gpu(args.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_arc_dict = {'cifar10': 'resnet20', 'imagenet': 'densenet', 'gtsrb': 'convnet_fc', 'mnist': 'Lenet'}
    config_path = '../configs/%s/train_configs/%s.yaml' % (args.dataset, model_arc_dict[args.dataset])
    conf = mlconfig.load(config_path)
    print(conf)

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# read dataset: %s " % args.dataset)
    train_data, test_data = load_init_data(dataname=args.dataset, device=device, download=args.download, dataset_path=args.datapath, config=conf)

    print("\n# construct poisoned dataset")
    # train_data_loader, test_data_ori_loader, test_data_tri_loader, normalizer = create_backdoor_data_loader(args.dataset, train_data, test_data, args.trigger_label, args.poisoned_portion, args.batchsize, device, mark_dir=args.mark_dir, alpha=args.alpha, task=args.task)
    train_data_loader, test_data_ori_loader, test_data_tri_loader, normalizer = create_backdoor_data_loader(
        args.dataset, train_data, test_data, args.trigger_label, args.poisoned_portion, args.batchsize, device,
        mark_dir=args.mark_dir, alpha=args.alpha, task=args.task, targeted_attack=args.vulnerable_targeted_attack)

    # validate(test_data_ori_loader, model_target, torch.nn.CrossEntropyLoss().cuda(), eval=True, one_hot=True)

    print("\n# begin training %s model" % args.task)
    if args.task == 'data_poison':
        attack_format = '%s_%s_%s' % (
        args.task, args.poison_level, 'generic' if args.trigger_label is None else 'specific')
    else:
        attack_format = '%s_normal' % args.task

    os.makedirs("../outputs/%s/attacks/%s" % (args.dataset, attack_format), exist_ok=True)

    for model_id in range(args.num):
        print('----------------------training %d-th %s model-------------------------' % (model_id, attack_format))
        backdoor_model_path = "../outputs/%s/attacks/%s/best_%d.pth" % (args.dataset, attack_format, model_id)
        # if not os.path.exists(backdoor_model_path):
        if not os.path.exists(backdoor_model_path):
            model = backdoor_model_trainer(
                    dataname=args.dataset,
                    train_data_loader=train_data_loader,
                    test_data_ori_loader=test_data_ori_loader,
                    test_data_tri_loader=test_data_tri_loader,
                    trigger_label=args.trigger_label,
                    epoch=args.epoch,
                    batch_size=args.batchsize,
                    loss_mode=args.loss,
                    optimization=args.optim,
                    lr=args.learning_rate,
                    print_perform_every_epoch=args.pp,
                    basic_model_path= backdoor_model_path,
                    device=device,
                    normalizer = normalizer,
                    minimal_update=True,
                    task = args.task
                    )
            print("\n# evaluation")
            print("## original test data performance:")
            print_model_perform(model, test_data_ori_loader, one_hot=False)
            print("## triggered test data performance:")
            if args.task == 'data_poison':
                poisoned_x, poisoned_y = train_data_loader.dataset.poison_x, train_data_loader.dataset.poison_y
                if args.dataset != 'gtsrb':
                    mean, std = get_mean_std(args.dataset)
                    poisoned_x = (poisoned_x - mean) / std
                poisoned_x = torch.tensor(poisoned_x, dtype=torch.float).cuda()
                poisoned_y = torch.from_numpy(np.array(poisoned_y)).cuda()
                test_data = TensorDataset(poisoned_x, poisoned_y)
                test_data_tri_loader_ = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
                print_model_perform(model, test_data_tri_loader_, one_hot= False)
            else:
                print_model_perform(model, test_data_tri_loader, one_hot= False)
        else:
            print('%s completed.' % backdoor_model_path)


if __name__ == "__main__":
    main()
