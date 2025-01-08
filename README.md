## BIV BenchMark


![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic)
![PyTorch 1.10](https://img.shields.io/badge/torch-1.10.1-green.svg?style=plastic)
![cuDNN 11.3](https://img.shields.io/badge/cudnn-11.3-green.svg?style=plastic)

## Get Started
We describe a manual installation and usage. First, install all dependencies via conda.
```shell
$ conda env create -f environment.yaml
```

The following five main scripts provide the entire toolbox's functionality:

- *train_protect_model.py*: Trains a neural network to be protected.
- *train_tampered_model.py*: runs basic tampering attacks on a pre-trained model.
- *backdoor_attack/main.py*: Runs data poisoning-based attacks on a pre-trained model.
- *data_removal/Unlearning.py*: Runs an unlearning attack on a pre-trained model.
- *data_removal/Online_learning.py*: Runs an online learning attack on a pre-trained model.

We use the [mlconfig](https://github.com/narumiruna/mlconfig) library to pass configuration hyperparameters to each script. 
Examples of configuration files for CIFAR-10, GTSRB, MNIST, and ImageNet can be found in the ``configs/`` directory. 
Configuration files store **all hyperparameters** needed to reproduce an experiment. 

### Step 1: Configuring your protect model 

(a) Put the structure of your model in the `wrt/training/models/classifier` directory. 

(b) Put the training configuration of your model in the `configs/dataset_name/train_configs` directory.

Next, we use the resnet20 on CIFAR10 as an example. 


### Step 2: Training the protect model
Here, we use the resnet20 on CIFAR10 as the protect model. Running the following command to train the protect model.
```shell
$ python train_protect_model.py --config configs/cifar10/train_configs/resnet20.yaml
```

### Step 3: Tampering the pre-trained protect model
Running the following command to tamper the pre-trained protect model.
```shell
$ python train_tampered_model.py --attack_config configs/cifar10/attack_configs/ftll.yaml \
                                 --config configs/cifar10/train_configs/resnet20.yaml \
                                 --dataset cifar10 \
                                 --lr 1e-4
``` 

Note that the above command only support several basic tampering methods including *fine-tuning all layers (FTAL), 
fine-tuning the last layers(FTLL), freezing the last layers(FZLL), retraining all layers(Retraining ALL), retraining the last layers(Retraining Last), 
knowledge distillation, weight quantization, and weight pruning*. All configurations are stored in the `configs/dataset_name/attack_configs` directory. You can modify the configuration file to run different tampering methods.

If you want to use other tampering methods, 
you can follow below instructions.

#### (a) Data poison-based attacks
These attacks including *random class degradation attack (Degradation<sub>Random</sub>-C), 
specific class degradation attack (Degradation<sub>Specific</sub>-C),
random sample degradation attack (Degradation-S), clean label backdoor attack (Clean Label),
neuron trojan attack(Trojan attack) and
targeted attack*. 

All these attacks are supported in our repository based on [backdoor attack](https://github.com/vtu81/backdoor_attack) and are in the `backdoor_attack` directory. To run the attack, you can use the following command:
```shell
$ cd backdoor_attack
$ python backdoor_attack/main.py --dataset cifar10 --task backdoor 
--poison_level class --trigger_label 0 --poisoned_portion 0.1 
--epoch 3 --gpu 2 --vulnerable_targeted_attack False --optim adam 
--batchsize 64 --num 1 --learning_rate 0.001
```

#### (b) data removal based attacks
These attacks include *unlearning-based attacks and online learning-based attacks*.

##### (b.1) Unlearning based attacks
Unlearning-based attacks refer to the process of forgetting the knowledge of the target model.  Below is an example of unlearning-based attacks on the CIFAR10 dataset.
```shell
$ cd data_remove
$ python data_remove/Unlearning.py --config ../configs/cifar10/train_configs/resnet20.yaml 
--dataset cifar10 --ratio 0.01  --batch_size 64 
--Posterior_Attack True --gpu 1 --num 10
```
##### (b.2) Online learning-based attacks
Online learning-based attacks refer to the process of learning the new knowledge of the target model.
Below is an example of online learning-based attacks on the CIFAR10 dataset.
```shell
$ cd data_remove
$ python data_remove/Online_learning.py --config ../configs/cifar10/train_configs/resnet20.yaml 
--dataset cifar10 --batch_size 128 --Posterior_Attack False 
--gpu 1 --fp_path ../fingerprints.npy --num 10 
```


## Datasets
Our BIV Bench currently implements custom data loaders (*class WRTDataLoader*) for the following datasets. 

- MNIST
- CIFAR-10
- ImageNet (needs manual download)
- Omniglot (needs manual download)
- Open Images (needs manual download)
- GTSRB (needs manual download)


## Reference
The codebase is based on [Watermark-Robustness-Toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox) and 
[backdoor attack](https://github.com/vtu81/backdoor_attack).
## Cite our paper
```
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




