import os.path
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from wrt.utils import *
from train_tampered_model import file_with_suffix_exists, __load_model
import mlconfig
from utils_torch import *
from Unlearning import evaluate, MI_attack

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/cifar10/train_configs/resnet20.yaml',
                        help="Path to config file. Determines all training params.")
    parser.add_argument('-w', "--RD_models_dir", type=str,
                        default="../outputs/cifar10/random_start/resnet20",
                        help="Path to the directory with the random start model."
                             "This scripts expects a 'best.pth' and one '*.yaml' file "
                             "to exist in this dir.")
    parser.add_argument('--dataset',  default='cifar10')
    parser.add_argument('-r', "--resume", type=str,
                        default=None,
                        help="Path to checkpoint to continue the attack. ")
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--ratio',  default=0.01, help='default forget one batch')
    parser.add_argument("--num", type=int, default=10, help="num of training models, default: 10")
    parser.add_argument("--content_constraint", type=bool, default=False, help="whether to use content constraint")
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--Posterior_Attack', default=False)
    parser.add_argument('--coeff_negative_gradient', default=0.1)
    parser.add_argument("--gpu", type=str, default='2', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument('--datapath', default='backdoor_attack/dataset/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--fp_path', default='..fingerprints.npy', help='Path to fingerprints')

    # parser.add_argument('--num', type=int, default=10, help='num of training models, default: 10')

    return parser.parse_args()


def get_protect_model(args, config):
    model = config.model(normalize=True).cuda()
    if args.dataset == 'imagenet':
        protect_model = model
        protect_model.eval()
    else:
        protect_model = load_protect_model(args.dataset, model, normalize=False)
    return protect_model


if __name__ == "__main__":
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_args()
    reserve_gpu(args.gpu)
    config = mlconfig.load(args.config)
    print(config)
    # load protect model
    protect_model = get_protect_model(args, config)

    # evaluate protect model
    train_loader = config.dataset(train=True, normalization=False, apply_augmentation=True)
    valid_loader = config.dataset(train=False, normalization=False, apply_augmentation=False)
    validate(valid_loader, protect_model, torch.nn.CrossEntropyLoss().cuda())

    # load and prepare fingerprint
    ssf = False
    fingerprints = np.load(args.fp_path)  # load_fingerprints(args, ssf, config)
    fingerprints = torch.from_numpy(fingerprints).cuda()
    y_fp = protect_model(fingerprints).argmax(dim=-1).cpu().detach().numpy()

    np.random.seed(123)
    os.makedirs('../outputs/%s/attacks/online_learning' % args.dataset, exist_ok=True)

    latent_fp = protect_model(fingerprints, feature=True).cpu().detach().numpy()
    plot_kendall_rank(latent_fp, plot_ratio=1.0)

    # load initial data
    from backdoor_attack.data import load_init_data
    train_data, test_data = load_init_data(dataname=args.dataset, device=device, download=True,
                                           dataset_path=args.datapath)
    x_train = np.moveaxis((train_data.data / 255.), source=-1, destination=1)
    x_test = np.moveaxis((test_data.data / 255.), source=-1, destination=1)
    y_train, y_test = np.array(train_data.targets), np.array(test_data.targets)

    # random select unseen data points to train
    num_to_train = 1   # args.batch_size  # int(len(x_train) * args.ratio)
    root_path = '../outputs/%s/attacks/online_learning/' % args.dataset
    data_unseen_path = root_path + 'unseen_data_%d.npy' % num_to_train
    label_unseen_path = root_path + 'unseen_label_%d.npy' % num_to_train
    if os.path.exists(data_unseen_path):
        data_unseen = np.load(data_unseen_path)
        label_unseen = np.load(label_unseen_path)
    else:
        # select a not confident data
        classifier = load_protect_model(args.dataset, config.model().cuda(), eval=True)
        x_t = torch.tensor(x_test, dtype=torch.float32).cuda()
        idx_not_conf = torch.softmax(classifier(x_t), dim=-1).max(dim=-1)[0] < 0.8
        idx_correct = classifier(x_t).argmax(dim=-1) == torch.from_numpy(y_test).cuda()
        idx_correct_not_conf = idx_not_conf & idx_correct
        if torch.sum(idx_correct_not_conf) != 0.:
            index_unseen = np.array([idx for idx in range(len(x_t)) if idx_correct_not_conf[idx]])[:num_to_train]
        else:
            index_unseen = np.random.choice(a=range(len(x_test)), size=num_to_train, replace=False)
        data_unseen, label_unseen = x_test[index_unseen], y_test[index_unseen]
        np.save(data_unseen_path, data_unseen)
        np.save(label_unseen_path, label_unseen)


    # Online learning the selected data
    epochs = 10
    criterion = nn.CrossEntropyLoss().cuda()
    inputs_unseen = torch.tensor(data_unseen, dtype=torch.float32).cuda()
    labels_unseen = torch.from_numpy(label_unseen).cuda()
    for model_id in range(args.num):
        # model path
        model = config.model(normalize=True).cuda()
        model_path = root_path + 'best_%d_%d.pth' % (model_id, num_to_train)

        # flag for early break
        early_break = False

        if args.Posterior_Attack:
            if os.path.exists(model_path):
                classifier = load_protect_model(args.dataset, model, normalize=False, eval=True)
                updated_model = config.model().cuda()
                updated_model.load_state_dict(torch.load(model_path))
                print('Eval membership inference attack from %s' % model_path)
                MI_attack(x_train, x_test, data_unseen, classifier=model, updated_model=updated_model)

            else:
                print('%s model not ready' % model_path)

        if not os.path.exists(model_path):
        # if True:
            protect_model = get_protect_model(args, config)
            # optimizer = config.optimizer(protect_model.parameters(), lr=1e-5)
            optimizer = torch.optim.SGD(protect_model.parameters(), lr=1e-4)

            best_acc = - np.inf
            print('-------------training %d-th model--------------' % model_id)
            protect_model.eval()  # freeze the BN layer and dropout layer for fine-tuning
            epochs = 1
            for e in range(epochs):
                if early_break: break
                # online learning
                for batch_id, (x, y) in enumerate(train_loader):
                    inputs = x.cuda()
                    labels = y.cuda()
                    outputs = protect_model(inputs)
                    loss_normal = criterion(outputs, labels)
                    outputs_unseen = protect_model(inputs_unseen)
                    loss_ng = criterion(outputs_unseen, labels_unseen)

                    # maintain the normal performance while increase the performance on marked data points
                    grad_normal = torch.autograd.grad(loss_normal, protect_model.parameters())
                    grad_ng = torch.autograd.grad(loss_ng, protect_model.parameters())
                    for i, (para_name, para) in enumerate(protect_model.named_parameters()):
                        para.requires_grad = False

                        # gradient projection
                        grad_ng_proj = grad_ng[i] - torch.bmm((torch.bmm(grad_ng[i].view(1, 1, -1),
                                                                         grad_normal[i].view(1, -1, 1)))
                                                              / (1e-20 + torch.bmm(
                            grad_normal[i].view(1, 1, -1),
                            grad_normal[i].view(1, -1, 1)))
                                                              .view(1, 1, 1),
                                                              grad_normal[i].view(1, 1, -1)).view(
                            grad_ng[i].shape)

                        # parameter update with projected gradient
                        para -= 1e-4 *(grad_normal[i] + grad_ng_proj)
                        para.requires_grad = True

                    # optimizer.zero_grad()
                    # loss = loss_normal + loss_ng
                    # loss.backward()
                    # optimizer.step()

                    # eval for early break
                    with torch.no_grad():
                        protect_model.eval()
                        val_acc = validate(valid_loader, protect_model, criterion=criterion)
                        print('epoch %d, step %d, val acc %f' % (e, batch_id, val_acc))
                        confidence = torch.softmax(protect_model(inputs_unseen[:num_to_train]), dim=-1).max(dim=-1)[
                            0]
                        confident = confidence > 0.95
                        correct = torch.softmax(protect_model(inputs_unseen[:num_to_train]), dim=-1).argmax(dim=-1) \
                                  == labels_unseen[:num_to_train]
                        confidence = confidence if num_to_train <= 1 else confidence.mean()
                        print('confidence: %f, correct: %s' % (
                        confidence.cpu().detach().numpy(), str(correct.cpu().detach().numpy())))
                        if confident and correct:
                            early_break = True
                            print('early break at epoch: %d, step: %d' % (e, batch_id))
                            break


                        # # switch the model for training
                        # protect_model.train()

                # eval the normal performance
                acc = validate(valid_loader, protect_model, criterion=criterion)

                # save the best model
                if args.save:
                    if acc > best_acc:
                        best_acc = acc
                        torch.save(protect_model.state_dict(), model_path)

        # else:
        #     print('%s model ready' % model_path)

        # load the best model
        model.load_state_dict(torch.load(model_path))
        # eval the normal performance
        model.eval()
        validate(valid_loader, model, criterion=criterion)
        # eval the unseen performance
        confidence = torch.softmax(model(inputs_unseen[:num_to_train]), dim=-1).max(dim=-1)[0]
        correct = torch.softmax(model(inputs_unseen[:num_to_train]), dim=-1).argmax(dim=-1) \
                  == labels_unseen[:num_to_train]
        confidence = confidence if num_to_train <= 1 else confidence.mean()
        print('performance of newly learned sample confidence: %f, correct classification: %s' % (
            confidence.cpu().detach().numpy(), str(correct.cpu().detach().numpy())))

        # eval fingerprints detection performance
        y_hat = model(fingerprints).argmax(dim=-1).cpu().detach().numpy()
        print('fingerprints detection rates: %f' % (y_hat != y_fp).mean())