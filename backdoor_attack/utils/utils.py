import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from PIL import Image
import numpy as np 
import pandas as pd
import torch.nn.functional as F
import GPUtil as GPUtil
import time
from torchvision import transforms
import math
from tqdm import tqdm




def array2img(x):
    a = np.array(x)
    img = Image.fromarray(a.astype('uint8'), 'RGB')
    img.show()


def print_model_perform(model, data_loader, one_hot=True):
    model.eval() # switch to eval mode
    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_predict.append(batch_y_predict)
        if one_hot:
            batch_y = torch.argmax(batch_y, dim=1)
        y_true.append(batch_y)

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    if one_hot:
        # try:
        #     target_names_idx = set.union(set(np.array(y_true.cpu())), set(np.array(y_predict.cpu())))
        #     target_names = [data_loader.dataset.classes[i] for i in target_names_idx]
        #     print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=target_names))
        # except ValueError as e:
            print('acc: %f' % (y_true.argmax(dim=-1) == y_predict).cpu().detach().numpy().mean())
            # print(e)
    else:
        if len(y_true.shape) > 1:
            print('acc: %f' % (y_true.argmax(dim=-1) == y_predict).cpu().detach().numpy().mean())
        else:
            print('acc: %f' % (y_true == y_predict).cpu().detach().numpy().mean())


def select_vulnerable_sample(num_classes, model_target, x_train, x_test, y_train, y_test, k=10, normalizer=None):
    """

    Args:
        num_classes:  num of classes
        model_target:
        x_train:
        x_test:
        y_train:
        y_test:
        k:
        target_layer:

    Returns:

    """
    model_target.eval()
    y_pre_test = F.softmax(model_target(normalizer(x_test)), dim=-1)
    max_prob = y_pre_test.max(axis=-1)[0]

    correct_idx_test = (y_pre_test.argmax(axis=-1) == y_test) & (max_prob > torch.quantile(max_prob, 0.5))

    x_test = x_test[correct_idx_test]
    y_test = y_test[correct_idx_test]
    y_pre, y_true = model_target(normalizer(x_train)).argmax(axis=-1), y_train
    correct_idx = (y_pre == y_true)
    latent_x = model_target(normalizer(x_train[correct_idx]), latent=True)


    latent_centroid = np.array([latent_x[y_true[correct_idx] == y].mean(axis=0).detach().cpu().numpy()
                                for y in range(num_classes)])
    latent_test = model_target(normalizer(x_test), latent=True).detach().cpu().numpy()
    l2_dist = np.linalg.norm(latent_test[:, np.newaxis, :] - latent_centroid, axis=-1)
    y_test =  np.eye(num_classes)[y_test.detach().cpu().numpy()].astype(np.bool) # to one hot vector
    delta_distance = l2_dist[y_test] - l2_dist[~y_test].reshape((y_test.shape[0], -1)).min(axis=-1)
    _, idx_selected = find_topk(a=delta_distance, k=k, axis=-1, largest=True, sorted=True)  # delta_distance.argmax()
    x_test_selected = x_test[idx_selected]
    y_target = l2_dist[idx_selected][~y_test[idx_selected]].reshape((k, -1)).argmin(axis=-1)
    y_target[y_target >= y_test[idx_selected].argmax(axis=-1)] += 1

    return x_test_selected.detach().cpu().numpy(), y_target


def select_base_and_target_instance(dataset, base_label, target_label):
    base_instance = None
    target_instance = None

    for inputs, labels in dataset:
        # for i in range(inputs.shape[0]):
        if labels == base_label:  # if it's a base class
            base_instance = inputs.transpose(2, 0, 1) # torch.from_numpy(inputs).unsqueeze(0).cuda()
        elif labels == target_label:  # if it's a target class
            target_instance = inputs.transpose(2, 0, 1) # torch.from_numpy(inputs).unsqueeze(0).cuda()

        # if both instance are enough, return
        if base_instance is not None and target_instance is not None:
            return torch.tensor(base_instance, dtype=torch.float).unsqueeze(0).cuda(), torch.tensor(target_instance, dtype=torch.float).unsqueeze(0).cuda()

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.detach().cpu().numpy().transpose((1, 2, 0))
    # # undo image normalization
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # input = std * input + mean
    # input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

def optimize_poison_instance(base_instance, target_instance, model, transforms_normalization, verbose=False):
    model.eval()
    perturbed_instance = base_instance.clone()
    target_features = model(target_instance, latent=True)

    epsilon = 8 / 255
    alpha = 0.05 / 255

    start_time = time.time()
    for i in range(1000):
        perturbed_instance.requires_grad = True

        poison_instance = transforms_normalization(perturbed_instance)
        poison_features = model(poison_instance, latent=True)

        feature_loss = torch.nn.MSELoss()(poison_features, target_features)
        image_loss = torch.nn.MSELoss()(poison_instance, base_instance)
        loss = feature_loss # + image_loss / 1e2
        loss.backward(retain_graph=True)

        signed_gradient = perturbed_instance.grad.sign()

        perturbed_instance = perturbed_instance - alpha * signed_gradient
        eta = torch.clamp(perturbed_instance - base_instance, -epsilon, epsilon) # clip to [-epsilon, epsilon]
        perturbed_instance = torch.clamp(base_instance + eta, 0, 1).detach()

        if i == 0 or (i + 1) % 500 == 0:
            print(f'Feature loss: {feature_loss}, Image loss: {image_loss}, Time: {time.time() - start_time}')
    if verbose:
        imshow(base_instance[0].cpu(), 'Original Instance')
        imshow(target_instance[0].cpu(), 'Target Instance')
        imshow(poison_instance[0].cpu(), 'Poison Instance')
    return perturbed_instance.detach().cpu().numpy()



def tanh_func(x: torch.Tensor) -> torch.Tensor:
    r"""tanh object function.
    .. code-block:: python
        return x.tanh().add(1).mul(0.5)
    Args:
        x (torch.Tensor): The tensor ranging from
            :math:`[-\infty, +\infty]`.
    Returns:
        torch.Tensor: The tensor ranging in ``[0, 1]``
    """
    return x.tanh().add(1).mul(0.5)


def atan_func(x: torch.Tensor) -> torch.Tensor:
    r"""arctan object function.
    .. code-block:: python
        return x.atan().div(math.pi).add(0.5)
    Args:
        x (torch.Tensor): The tensor ranging from
            :math:`[-\infty], +\infty]`.
    Returns:
        torch.Tensor: The tensor ranging in ``[0, 1]``
    """
    return x.atan().div(math.pi).add(0.5)

def optimize_trojan_trigger(model, dataset_name, trigger_size=4, neuron_num=1, verbose=False,
                            img_shape=(3, 32, 32), target_value=100, iter_steps=1000):
    # a.calculate most likely max-activated neuron for trojan
    preprocess_next_layer_dict = {'mnist': 'fc3.weight', 'cifar10': 'fc.weight',
                                  'gtsrb': 'fc.weight', 'imagenet': 'fc.weight'}

    preprocess_next_layer = preprocess_next_layer_dict[dataset_name]
    weight = model.state_dict()[preprocess_next_layer].abs()
    if weight.dim() > 2:
        weight = weight.flatten(2).sum(2)
    Max_connected_kernel = weight.sum(0).argsort(descending=True)[:neuron_num]

    # b.initialize trigger and mask
    mask = np.zeros(img_shape)
    mask[:, -trigger_size:, -trigger_size:] = 1.0

    # c.optimize the trigger by max activating the output of target kernel
    mask_tensor = torch.tensor(mask, dtype=torch.float32).cuda()
    mask_tensor.requires_grad = False
    mask_mask = (~ mask_tensor.bool()).int()
    mask_mask.requires_grad = False
    model.eval()

    background = torch.zeros(img_shape).unsqueeze(0).cuda()
    atanh_pattern_tensor = torch.randn_like(background, requires_grad=True)

    optimizer = torch.optim.Adam(params=[atanh_pattern_tensor], lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter_steps)
    avg_losses = []

    print('optimizing trigger...')
    for step in tqdm(range(iter_steps)):
        x_input = background * mask_mask + tanh_func(atanh_pattern_tensor) * mask_tensor
        latent = model(x_input, latent=True).squeeze()
        output = latent[Max_connected_kernel].abs()
        if output.dim() > 2:
            output = output.flatten(2).sum(2)
        loss = F.mse_loss(output, target_value * torch.ones_like(output).cuda(), reduction='mean')
        optimizer.zero_grad()
        loss.backward(inputs=[atanh_pattern_tensor])
        avg_losses.append(loss.item())
        # x_input.grad.data.mul_(mask_tensor)
        optimizer.step()
        lr_scheduler.step()

    plt.imshow(x_input[0].detach().cpu().numpy().transpose(1, 2, 0))
    os.makedirs('outputs/%s' % dataset_name, exist_ok=True)
    plt.savefig('outputs/%s/trojan_trigger.png' % dataset_name)
    if verbose:
        plt.show()

    # return trigger
    return  x_input.detach().cpu().numpy().squeeze()


def find_topk(a, k, axis=-1, largest=True, sorted=True):
    if axis is None:
        axis_size = a.size
    else:
        axis_size = a.shape[axis]
    assert 1 <= k <= axis_size

    a = np.asanyarray(a)
    if largest:
        index_array = np.argpartition(a, axis_size-k, axis=axis)
        topk_indices = np.take(index_array, -np.arange(k)-1, axis=axis)
    else:
        index_array = np.argpartition(a, k-1, axis=axis)
        topk_indices = np.take(index_array, np.arange(k), axis=axis)
    topk_values = np.take_along_axis(a, topk_indices, axis=axis)
    if sorted:
        sorted_indices_in_topk = np.argsort(topk_values, axis=axis)
        if largest:
            sorted_indices_in_topk = np.flip(sorted_indices_in_topk, axis=axis)
        sorted_topk_values = np.take_along_axis(
            topk_values, sorted_indices_in_topk, axis=axis)
        sorted_topk_indices = np.take_along_axis(
            topk_indices, sorted_indices_in_topk, axis=axis)
        return sorted_topk_values, sorted_topk_indices
    return topk_values, topk_indices


def get_max_index(data_dir, suffix):
    """ Lists all files from a folder and checks the largest integer prefix for a filename (in snake_case) that
    contains the suffix
    """
    index = 0
    for filename in os.listdir(data_dir):
        if suffix in filename:
            index = int(filename.split("_")[0]) + 1 if int(filename.split("_")[0]) >= index else index
    return str(index)


def pick_gpu():

    """
    Picks a GPU with the least memory load.
    :return:
    """
    try:
        gpu = GPUtil.getFirstAvailable(order='memory', maxLoad=2, maxMemory=0.8, includeNan=False,
                                       excludeID=[], excludeUUID=[])[0]
        return gpu
    except Exception as e:
        print(e)
        return "0"


def reserve_gpu(mode_or_id):
    """ Chooses a GPU.
    If None, uses the GPU with the least memory load.
    """
    if mode_or_id:
        gpu_id = mode_or_id
        os.environ["CUDA_VISIBLE_DEVICES"] = mode_or_id
    else:
        gpu_id = str(pick_gpu())
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Selecting GPU id {gpu_id}")



