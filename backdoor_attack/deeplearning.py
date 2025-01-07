import os
import numpy as np 
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from models import BadNet
# from utils import print_model_perform
import mlconfig

import sys
sys.path.append("/home/hdd/baixiaofan/PycharmProjects/BIV_Bench")
from utils_torch import load_protect_model


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer


def backdoor_model_trainer(dataname, train_data_loader, test_data_ori_loader, test_data_tri_loader, trigger_label,
                           epoch, batch_size, loss_mode, optimization, lr, print_perform_every_epoch,
                           basic_model_path, device, normalizer=None, minimal_update=False, task='backdoor'):

    clean_label_backdoor_training = (task == 'clean_label_backdoor')
    if dataname in ['cifar10', 'imagenet', 'gtsrb', 'mnist']:
        model_arc_dict = {'cifar10': 'resnet20', 'imagenet': 'densenet', 'gtsrb': 'convnet_fc', 'mnist': 'Lenet'}
        config_path = '../configs/%s/train_configs/%s.yaml' % (dataname, model_arc_dict[dataname])
        conf = mlconfig.load(config_path)
        print(conf)
        badnet = conf.model().to(device)
        # initialize model with pre-trained protect model
        badnet = load_protect_model(dataname, badnet, normalize=False)
    else:
        badnet = BadNet(input_channels=train_data_loader.dataset.channels, output_num=train_data_loader.dataset.class_num).to(device)


    criterion = loss_picker(loss_mode)
    optimizer = optimizer_picker(optimization, badnet.parameters(), lr=lr)
    acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='test', print_perform=False,
                        normalizer=None)
    print("### ori test acc is %.4f" % acc_test_ori)

    train_process = []
    print("### target label is %s, EPOCH is %d, Learning Rate is %f" % (str(trigger_label), epoch, lr))
    print("### Train set size is %d, ori test set size is %d, tri test set size is %d\n" %
          (len(train_data_loader.dataset), len(test_data_ori_loader.dataset), len(test_data_tri_loader.dataset)))

    if task in ['trojan']:
        # set only the last layer to be trainable in clean label backdoor training for badnet

        # Freeze all layers
        for param in badnet.parameters():
            param.requires_grad = False

        # Unfreeze the last layer
        last_layer = list(badnet.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True

        # flag to only train the last layer, filter out the parameters that do not require gradients
        optimizer = optimizer_picker(optimization, filter(lambda p: p.requires_grad, badnet.parameters()),
                                     lr=lr)
    else:
        pass

    for epo in range(epoch):


        if not minimal_update:
            loss, break_flag = train(badnet, train_data_loader, criterion, optimizer, loss_mode)
        else:
            loss, break_flag = train_with_minimal_update(badnet, train_data_loader, criterion, optimizer, loss_mode,
                                                         test_loader=test_data_tri_loader)

        acc_train = eval(badnet, train_data_loader, batch_size=batch_size, mode='backdoor',
                         print_perform=print_perform_every_epoch)
        acc_test_ori = eval(badnet, test_data_ori_loader, batch_size=batch_size, mode='backdoor',
                            print_perform=print_perform_every_epoch)
        acc_test_tri = eval(badnet, test_data_tri_loader, batch_size=batch_size, mode='backdoor',
                            print_perform=print_perform_every_epoch)
        print("# EPOCH%d   loss: %.4f  training acc: %.4f, ori testing acc: %.4f, trigger testing acc: %.4f\n" \
              % (epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        if break_flag:
            print("### Training is done, trigger acc is %.4f" % acc_test_tri)
            break

        # save model 
        torch.save(badnet.state_dict(), basic_model_path)

        # save training progress
        train_process.append(( dataname, batch_size, trigger_label, lr, epo, loss.item(), acc_train, acc_test_ori, acc_test_tri))
        df = pd.DataFrame(train_process, columns=("dataname", "batch_size", "trigger_label", "learning_rate", "epoch", "loss", "train_acc", "test_ori_acc", "test_tri_acc"))
        df.to_csv("./logs/%s_train_process_trigger%s.csv" % (dataname, str(trigger_label)), index=False, encoding='utf-8')

    return badnet


def train_with_minimal_update(model, data_loader, criterion, optimizer, loss_mode, test_loader):
    running_loss = 0
    model.train()
    # loop = tqdm(enumerate(data_loader))
    loop = enumerate(data_loader)
    for step, (batch_x, batch_y) in loop:
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross_entropy":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss

        # every 10 steps, evaluate the model
        if step % 10 == 0:
            acc_tri = eval(model, test_loader, batch_size=64, mode='backdoor', print_perform=True, normalizer=None)
            if acc_tri > 0.9:
                print('early break at step %d, acc_tri is %.4f' % (step, acc_tri))
                break
    return running_loss, True


def train(model, data_loader, criterion, optimizer, loss_mode):
    running_loss = 0
    model.train()
    # loop = tqdm(enumerate(data_loader))
    loop = enumerate(data_loader)
    for step, (batch_x, batch_y) in loop:
        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y) # mse loss
        elif loss_mode == "cross_entropy":
            loss = criterion(output, torch.argmax(batch_y, dim=1)) # cross entropy loss

        loss.backward()
        optimizer.step()
        running_loss += loss

    return running_loss, False


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, normalizer=None):
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    # loop = tqdm(enumerate(data_loader))
    loop = enumerate(data_loader)
    if normalizer is None:
        for step, (batch_x, batch_y) in loop:
            batch_y_predict = model(batch_x)
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_predict.append(batch_y_predict)

            batch_y = torch.argmax(batch_y, dim=1)
            y_true.append(batch_y)
    else:
        for step, (batch_x, batch_y) in loop:
            batch_y_predict = model(normalizer(batch_x))
            batch_y_predict = torch.argmax(batch_y_predict, dim=1)
            y_predict.append(batch_y_predict)

            batch_y = torch.argmax(batch_y, dim=1)
            y_true.append(batch_y)
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if print_perform and mode is not 'backdoor':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))
    # else:
    #     print(accuracy_score(y_true.cpu(), y_predict.cpu()))
    #     print((y_true.cpu()==y_predict.cpu()).sum() / len(y_true))

    return accuracy_score(y_true.cpu(), y_predict.cpu())

