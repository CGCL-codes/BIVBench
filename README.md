# BIV Benchmark

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.10](https://img.shields.io/badge/torch-1.10.1-green.svg?style=plastic)
![cuDNN 11.3](https://img.shields.io/badge/cudnn-11.3-green.svg?style=plastic)

## Overview

The BIV Benchmark provides a comprehensive toolbox for evaluating black-box integrity verification of deep neural network models. It includes scripts for training, tampering, and attacking models, as well as for implementing various data removal techniques. This repository also supports multiple datasets and allows for easy configuration of experiments using YAML files.

---

## Installation

### Step 1: Install Dependencies
Use the provided `environment.yaml` file to create the required Conda environment:
```shell
conda env create -f environment.yaml
```

## Available Scripts

The toolbox includes the following main scripts:

- **`train_protect_model.py`**: Trains a neural network to serve as the protected model.
- **`train_tampered_model.py`**: Performs basic tampering attacks on a pre-trained model.
- **`backdoor_attack/main.py`**: Executes data poisoning-based attacks on a pre-trained model.
- **`data_remove/Unlearning.py`**: Runs unlearning-based attacks on a pre-trained model.
- **`data_remove/Online_learning.py`**: Executes online learning-based attacks on a pre-trained model.

---

## Configuration

The toolbox uses the [mlconfig](https://github.com/narumiruna/mlconfig) library for passing configuration parameters. Example configuration files for datasets such as CIFAR-10, GTSRB, MNIST, and ImageNet are available in the `configs/` directory. These files store **all hyperparameters** needed to reproduce experiments.

## Getting Started

### Step 1: Configure the Protected Model

1. Add the model architecture to the `wrt/training/models/classifier` directory.
2. Add the training configuration file to the `configs/dataset_name/train_configs` directory.

---

### Step 2: Train the Protected Model

To train a model (e.g., ResNet-20 on CIFAR-10), use the following command:
```bash
python train_protect_model.py --config configs/cifar10/train_configs/resnet20.yaml
```

---

### Step3: Tamper with the Pre-Trained Protected Model

Run the following command to tamper with the pre-trained protected model:

```bash
python train_tampered_model.py --attack_config configs/cifar10/attack_configs/ftll.yaml \
                               --config configs/cifar10/train_configs/resnet20.yaml \
                               --dataset cifar10 \
                               --lr 1e-4
```

### Supported Tampering Methods

- **Fine-tuning all layers (FTAL)**
- **Fine-tuning the last layers (FTLL)**
- **Freezing the last layers (FZLL)**
- **Retraining all layers (Retraining ALL)**
- **Retraining the last layers (Retraining Last)**
- **Knowledge distillation**
- **Weight quantization**
- **Weight pruning**

Configurations for tampering methods are stored in the `configs/dataset_name/attack_configs` directory. Modify these files to use different tampering methods.

## Advanced Tampering Methods

### (a) Data Poisoning-Based Attacks

Supported attacks include:
- **Random class degradation (Degradation<sub>Random</sub>-C)**
- **Specific class degradation (Degradation<sub>Specific</sub>-C)**
- **Random sample degradation (Degradation-S)**
- **Clean label backdoor attacks**
- **Neuron trojan attacks**
- **Targeted attacks**

To perform a backdoor attack (e.g., CIFAR-10 dataset), use the following command:
```bash
cd backdoor_attack
python backdoor_attack/main.py --dataset cifar10 --task backdoor \
                               --poison_level class --trigger_label 0 \
                               --poisoned_portion 0.1 --epoch 3 --gpu 2 \
                               --vulnerable_targeted_attack False \
                               --optim adam --batchsize 64 --num 1 \
                               --learning_rate 0.001
```

## (b) Data Removal-Based Attacks

### (b.1) Unlearning-Based Attacks

Unlearning attacks remove knowledge from the target model. Example:
```bash
cd data_remove
python Unlearning.py --config ../configs/cifar10/train_configs/resnet20.yaml \
                     --dataset cifar10 --ratio 0.01 --batch_size 64 \
                     --Posterior_Attack True --gpu 1 --num 10
```
### (b.2) Online Learning-Based Attacks

Online learning attacks introduce new knowledge to the target model. Example:
```bash
cd data_remove
python Online_learning.py --config ../configs/cifar10/train_configs/resnet20.yaml \
                          --dataset cifar10 --batch_size 128 \
                          --Posterior_Attack False --gpu 1 \
                          --fp_path ../fingerprints.npy --num 10
```

## Supported Datasets

The following datasets are supported via custom data loaders (`class WRTDataLoader`):

- **MNIST**
- **CIFAR-10**
- **ImageNet** (requires manual download)
- **Omniglot** (requires manual download)
- **Open Images** (requires manual download)
- **GTSRB** (requires manual download)

---

## Reference

This codebase is built on:
- [Watermark-Robustness-Toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox)
- [backdoor attack](https://github.com/vtu81/backdoor_attack)


## Citation

Please cite our paper if you use this benchmark in your work:

```bibtex
@inproceedings{10.1145/3664647.3681691,
  author = {He, Chaoxiang and Bai, Xiaofan and Ma, Xiaojing and Zhu, Bin B. and Hu, Pingyi and Fu, Jiayun and Jin, Hai and Zhang, Dongmei},
  title = {Towards Stricter Black-box Integrity Verification of Deep Neural Network Models},
  year = {2024},
  isbn = {9798400706868},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3664647.3681691},
  doi = {10.1145/3664647.3681691},
  pages = {9875–9884},
  numpages = {10},
  location = {Melbourne VIC, Australia},
  series = {MM '24}
}
```
