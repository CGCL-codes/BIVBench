import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from backdoor_attack.data import load_init_data, get_mean_std, collect_n_samples


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
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--ratio',  default=0.01, help='default forget one batch')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--Posterior_Attack', default=True)
    parser.add_argument('--coeff_negative_gradient', default=0.1)
    parser.add_argument("--gpu", type=str, default='1', help="Which GPU to use. Defaults to GPU with least memory.")
    parser.add_argument('--datapath', default='backdoor_attack/dataset/', help='Place to load dataset (default: ./dataset/)')
    parser.add_argument('--num', type=int, default=10, help='num of training models, default: 10')

    return parser.parse_args()


def MI_attack(x_in, x_out, x_val, classifier, updated_model):
    classifier.eval()
    updated_model.eval()

    def batch_infer(x, model, batch_size=64):
        num_batch = int(np.ceil(len(x) / batch_size))
        start_index = 0
        y_pre = []
        for batch_i in tqdm(range(num_batch)):
            end_index = start_index + batch_size if batch_i < (num_batch - 1) else len(x)
            x_batch = torch.tensor(x[start_index:end_index], dtype=torch.float32).cuda()
            y_batch = torch.softmax(model(x_batch), dim=-1).cpu().detach().numpy()
            y_pre.append(y_batch)

            start_index = end_index

        y_pre = np.concatenate(y_pre, axis=0)
        return y_pre

    import sklearn.metrics as metrics
    x_temp = np.vstack((x_in, x_out))
    inv = np.ones((x_train.shape[0], 1))
    outv = np.zeros((x_test.shape[0], 1))
    in_out_temp = np.vstack((inv, outv))

    # sz = 2000
    # if sz > x_temp.shape[0]:
    #     sz = x_temp.shape[0]

    perm = np.random.permutation(x_temp.shape[0])
    # perm = perm[0:sz]
    x_targets = x_temp[perm, :]
    in_out_targets = in_out_temp[perm, :]
    pv = batch_infer(x_targets, model=classifier, batch_size=64)

    # search the best threshold
    best_acc = 0.0
    best_adv = 0.0
    best_threshold = 0.0
    for i in range(1, 100, 1):
        threshold = i / 100
        in_or_out_pred = np.zeros((x_targets.shape[0],))
        for i in range(len(pv)):
            largest_index = np.argmax(pv[i])
            if pv[i][largest_index] > threshold:
                in_or_out_pred[i] = 1
            else:
                in_or_out_pred[i] = 0

        cm = metrics.confusion_matrix(in_out_targets, in_or_out_pred)
        accuracy = np.trace(cm) / np.sum(cm.ravel())
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        advantage = tpr - fpr
        if advantage > best_adv:
            best_acc = accuracy
            best_adv = advantage
            best_threshold = threshold

    print('best_threshold: %f' % best_threshold)

    # attack
    p_val = batch_infer(x_val, model=updated_model, batch_size=64)
    attack_acc = (p_val.max(axis=-1) > best_threshold).sum() / len(p_val)
    print('Posterior membership inference attack acc: %f' % attack_acc)

def evaluate(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    loss = 0.0
    corrects = 0.0
    with torch.no_grad():
        loop = enumerate(dataloader)     # tqdm(enumerate(dataloader))
        for batch_i, (x, y) in loop:
            y_hat = model(x.cuda())
            val_loss = criterion(y_hat, y.cuda())
            _, val_preds = torch.max(y_hat, dim=-1)
            loss += val_loss.item()
            corrects += torch.sum(val_preds == y.cuda().data)
    return loss, corrects




if __name__ == "__main__":
    # Takes more time at startup, but optimizes runtime.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args = parse_args()
    reserve_gpu(args.gpu)
    config = mlconfig.load(args.config)
    print(config)
    np.random.seed(2024)
    root_path = 'outputs/%s/attacks/machine_unlearning/' % args.dataset
    os.makedirs(root_path, exist_ok=True)

    # load initial data
    train_loader = config.dataset(train=True, normalization=True)
    test_loader = config.dataset(train=False, normalization=True)
    x_train, y_train = collect_n_samples(len(train_loader.dataset), train_loader, has_labels=True,
                                         reduce_labels=True, verbose=False)
    x_test, y_test = collect_n_samples(len(test_loader.dataset), test_loader, has_labels=True,
                                       reduce_labels=True, verbose=False)

    # random select data to forget
    num_to_forget = args.num # args.batch_size  # int(len(x_train) * args.ratio)
    data_forget_path = root_path + 'forget_data.npy'
    label_forget_path = root_path + 'forget_label.npy'
    data_retain_path = root_path + 'retain_data.npy'
    label_retain_path = root_path + 'retain_label.npy'
    if os.path.exists(data_forget_path):
        data_forget = np.load(data_forget_path)
        label_forget = np.load(label_forget_path)
        x_train = np.load(data_retain_path)
        y_train = np.load(label_retain_path)
    else:
        data_forget, label_forget = x_train[:num_to_forget], y_train[:num_to_forget]
        x_train, y_train = x_train[num_to_forget:], y_train[num_to_forget:]
        np.save(data_forget_path, data_forget)
        np.save(label_forget_path, label_forget)
        np.save(data_retain_path, x_train)
        np.save(label_retain_path, y_train)

    # Unlearning the selected data by Negative Gradient
    epochs = 10
    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    # running_loss_history_ng = []
    # running_corrects_history_ng = []
    val_running_loss_history_ng = []
    val_running_corrects_history_ng = []

    criterion = nn.CrossEntropyLoss().cuda()

    inputs_ng = torch.tensor(data_forget, dtype=torch.float32).cuda()
    labels_ng = torch.from_numpy(label_forget).cuda()
    # print the confidence of inputs_ng before unlearning
    protect_model = load_protect_model(args.dataset, config.model().cuda(), normalize=False, eval=True)
    confidences = torch.softmax(protect_model(inputs_ng), dim=-1).max(dim=-1)[0]
    print('confidence of inputs_ng before unlearning: %s' % str(confidences))

    for model_id in range(args.num):
        # protect model path
        model = config.model().cuda()
        model_path = 'outputs/%s/attacks/machine_unlearning/best_%d.pth' % (args.dataset, model_id)
        # MI_attack(x_train, x_test, data_forget, classifier=load_protect_model('cifar10', model,normalize=False, eval=False))

        # flag for early break
        early_break = False

        if args.Posterior_Attack:
            if os.path.exists(model_path):
                # model.load_state_dict(torch.load(model_path))
                classifier = load_protect_model('cifar10', model,normalize=False, eval=True)
                updated_model = config.model().cuda()
                updated_model.load_state_dict(torch.load(model_path))
                print('Eval membership inference attack from %s' % model_path)
                # MI_attack(x_train, x_test, data_forget, classifier=model, updated_model=updated_model)
            else:
                print('%s model not ready' % model_path)

        # if not os.path.exists(model_path):
        if os.path.exists(model_path):
            protect_model = load_protect_model(args.dataset, model, normalize=False, eval=False)
            # optimizer = config.optimizer(protect_model.parameters(), lr=1e-4)
            optimizer = torch.optim.Adam(protect_model.parameters(), lr=1e-4)

            best_acc = - np.inf
            print('-------------training %d-th model--------------' % model_id)
            for e in range(10):
                if early_break: break

                running_loss = 0.0
                running_corrects = 0.0
                val_running_loss = 0.0
                val_running_corrects = 0.0

                # running_loss_ng = 0.0
                # running_corrects_ng = 0.0
                val_running_loss_ng = 0.0
                val_running_corrects_ng = 0.0

                # unlearning by Negative Gradient
                protect_model.eval()
                batch_num = int(np.ceil(len(x_train) / args.batch_size))
                start_idx = 0
                loop = range(batch_num)   # tqdm(range(batch_num))

                # for batch_id, (x, y) in enumerate(train_loader):
                #     y_hat = protect_model(x.cuda())
                #     loss_normal = criterion(y_hat, y.cuda())

                    # if inputs_ng is None:
                    #     idx = np.random.choice(a=range(len(x)), size=num_to_forget, replace=False)
                    #     inputs_ng, labels_ng = x[idx].cuda(), y[idx].cuda()
                    #
                    #     np.save()

                for batch_id in loop:
                    end_idx = start_idx + args.batch_size if batch_id < (batch_num - 1) else len(x_train)
                    inputs = torch.from_numpy(x_train[start_idx:end_idx]).cuda()
                    outputs = protect_model(inputs)

                    labels = torch.from_numpy(y_train[start_idx:end_idx]).cuda()
                    loss_normal = criterion(outputs, labels)

                    # unlearning by Negative Gradient for i-th inputs_ng
                    outputs_ng = protect_model(inputs_ng[model_id:model_id + 1])
                    loss_ng = criterion(outputs_ng, labels_ng[model_id:model_id + 1])

                    # # maintain the normal performance while decrease the performance on marked data points
                    #
                    # grad_normal = torch.autograd.grad(loss_normal, protect_model.parameters(), retain_graph=True)
                    # grad_ng = torch.autograd.grad(loss_ng, protect_model.parameters())

                    # learning_rate = 1e-4
                    # eps = 1e-8  # Numerical stability bias
                    #
                    # with torch.no_grad():
                    #     for i, (para_name, para) in enumerate(protect_model.named_parameters()):
                    #         # Flatten the gradient as a one-dimensional vector
                    #         grad_ng_flat = grad_ng[i].view(-1)
                    #         grad_normal_flat = grad_normal[i].view(-1)
                    #
                    #         # Calculate the gradient projection
                    #         dot_product = torch.dot(grad_ng_flat, grad_normal_flat)
                    #         norm_square = torch.dot(grad_normal_flat, grad_normal_flat) + eps
                    #         grad_ng_proj = grad_ng_flat - (dot_product / norm_square) * grad_normal_flat
                    #
                    #         # Restore the gradient shape
                    #         grad_ng_proj = grad_ng_proj.view(grad_ng[i].shape)
                    #
                    #         # Parameter updates
                    #         para -= learning_rate * grad_ng_proj

                    optimizer.zero_grad()
                    loss =   -  1 * loss_ng
                    loss.backward()
                    optimizer.step()

                    # _, preds = torch.max(outputs, 1)
                    # running_loss += loss_normal.item()
                    # running_corrects += torch.sum(preds == labels.data)
                    # start_idx = end_idx

                    # eval for early break
                    if ((batch_id + 1) % 1) == 0:
                        protect_model.eval()
                        val_loss_, val_corrects_ = evaluate(protect_model, test_loader)
                        val_loss_ = val_loss_ / len(x_test)
                        val_corrects_ = val_corrects_.float() / len(x_test)
                        print('epoch %d, step %d, val loss %f, val acc %f' % (e, batch_id + 1, val_loss_,
                                                                              val_corrects_), end='')
                        confidence = torch.softmax(protect_model(inputs_ng[model_id:model_id + 1]), dim=-1).max(dim=-1)[0]
                        print(', confidence %f' % confidence.cpu().detach().numpy())
                        protect_model.train()

                        if  confidence < 0.95:
                            early_break = True
                            print('early break at epoch: %d, step: %d' % (e, batch_id + 1))
                            break
                        # else:
                        #     # switch the model for training
                        #     protect_model.train()

                        if batch_id > 2:
                            early_break = True
                            print('Hard to forget. Early break at epoch: %d, step: %d' % (e, batch_id + 1))
                            break

                if early_break:
                    break

                # eval the normal performance
                protect_model.eval()
                with torch.no_grad():
                    batch_num_test = int(np.ceil(len(x_test) / args.batch_size))
                    start_idx = 0
                    for batch_id in range(batch_num_test):
                        end_idx = start_idx + args.batch_size if batch_id < (batch_num - 1) else len(x_test)
                        inputs = torch.tensor(x_test[start_idx:end_idx], dtype=torch.float32)
                        labels = torch.from_numpy(y_test[start_idx:end_idx])
                        val_inputs = inputs.cuda()
                        val_labels = labels.cuda()
                        val_outputs = protect_model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                        _, val_preds = torch.max(val_outputs, 1)
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(val_preds == val_labels.data)

                        start_idx = end_idx

                # eval the Negative Gradient Unlearning performance
                with torch.no_grad():
                    start_idx = 0
                    batch_ng = int(np.ceil(len(data_forget) / args.batch_size))
                    for batch_id in range(batch_ng):

                        end_idx = start_idx + args.batch_size if batch_id < (batch_ng - 1) else len(data_forget)
                        inputs = torch.tensor(data_forget[start_idx:end_idx], dtype=torch.float32)
                        labels = torch.from_numpy(label_forget[start_idx:end_idx])
                        val_inputs = inputs.cuda()
                        val_labels = labels.cuda()
                        val_outputs = protect_model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                        _, val_preds = torch.max(val_outputs, 1)
                        val_running_loss_ng += val_loss.item()
                        val_running_corrects_ng += torch.sum(val_preds == val_labels.data)

                        start_idx = end_idx


                # record normal epoch training acc
                val_loss_, val_corrects_ = evaluate(protect_model, test_loader)


                epoch_loss = running_loss / len(x_train)  / 2.
                epoch_acc = running_corrects / len(x_train) / 2.
                running_loss_history.append(epoch_loss)
                running_corrects_history.append(epoch_acc)

                # record normal epoch testing acc
                val_epoch_loss = val_running_loss / len(x_test)
                val_epoch_acc = val_running_corrects.float() / len(x_test)
                val_running_loss_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_acc)

                # # record Negative Gradient epoch training acc
                # epoch_loss_ng = running_loss_ng / num_to_forget
                # epoch_acc_ng = running_corrects_ng.float() / num_to_forget
                # running_loss_history_ng.append(epoch_loss_ng)
                # running_corrects_history_ng.append(epoch_acc_ng)

                # record Negative Gradient epoch testing acc
                val_epoch_loss_ng = val_running_loss_ng / num_to_forget
                val_epoch_acc_ng = val_running_corrects_ng.float() / num_to_forget
                val_running_loss_history_ng.append(val_epoch_loss_ng)
                val_running_corrects_history_ng.append(val_epoch_acc_ng)

                # save the best model
                if val_epoch_acc > best_acc:
                    best_acc = val_epoch_acc
                    torch.save(protect_model.state_dict(), 'outputs/%s/attacks/machine_unlearning/best_%d.pth' % (
                        args.dataset, model_id))

                print('epoch ', (e + 1))
                print('Normal training loss: {:.4f}, training acc {:.4f}'.format(epoch_loss, epoch_acc))
                print('Normal validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss, val_epoch_acc))

                # print('Unlearning training loss: {:.4f}, training acc {:.4f}'.format(epoch_loss_ng, epoch_acc_ng.item()))
                print('Unlearning validation loss: {:.4f}, validation acc {:.4f}'.format(val_epoch_loss_ng, val_epoch_acc_ng))