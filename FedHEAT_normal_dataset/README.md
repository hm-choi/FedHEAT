# Experimental Evaluation for CIFAR-10/100 and Tiny-ImageNet datasets

This directory contains the experimental code for the CIFAR-10/100 and Tiny-ImageNet datasets.

## Dataset
1. Download three generic object recognition datasets:
- [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

2. Extract the tgz or zip file into `./data/`

## Training
The commands below can be used to evaluate the federated learning methods.  
Users can freely specify the `exp_name` argument, while the recommended hyper-parameter settings for each dataset are provided below.

The `train.encrypted` option determines whether the experiment is performed in the plaintext or encrypted setting.

To select a dataset, set the `dataset` argument to one of the following:
- `cifar10`
- `cifar100`
- `tinyimagenet`

For example, the following command runs the evaluation on CIFAR-100: \
1. CIFAR-100, FedHEAT in encrypted version
~~~bash
CUDA_VISIBLE_DEVICES=0 python3 \
	federated_train.py \
	multiprocessing=True \
	main_gpu=0 \
	exp_name=FedAdam_ct_proposed_tau-in_10Minus5_encL3 \
	trainer=base \
	client=base \
	dataset=cifar100 \
	model=resnet18 \
	output_model_path=resnet18.pt \
	trainer.num_clients=100 \
	trainer.participation_rate=0.05 \
	server=FedAdam \
	trainer.encrypted=True \
	server.algo=proposed \
	server.tau_in=True \
	server.tau=0.00001 \
	trainer.enc_level=3 \
   server.t1=100 \
	server.t2=90 \
	split.alpha=0.3 \
	batch_size=50 \
	trainer.global_lr=0.01 \
	wandb=True
~~~

2. CIFAR-100, FedHEAT w/o MR in encrypted version
~~~bash
CUDA_VISIBLE_DEVICES=0 python3 \
	federated_train.py \
	multiprocessing=True \
	main_gpu=0 \
	exp_name=FedAdam_ct_original_tau-in_10Minus5_encL3 \
	trainer=base \
	client=base \
	dataset=cifar100 \
	model=resnet18 \
	output_model_path=resnet18.pt \
	trainer.num_clients=100 \
	trainer.participation_rate=0.05 \
	server=FedAdam \
	trainer.encrypted=True \
	server.algo=original \
	server.tau_in=True \
	server.tau=0.00001 \
	trainer.enc_level=3 \
	split.alpha=0.3 \
	batch_size=50 \
	trainer.global_lr=0.01 \
	wandb=True
~~~

3. CIFAR-100, HE-Adam in encrypted version
~~~bash
CUDA_VISIBLE_DEVICES=0 python3 \
	federated_train.py \
	multiprocessing=True \
	main_gpu=0 \
	exp_name=FedAdam_PPRL_ct_10Minus5_encL3 \
	trainer=base \
	client=base \
	dataset=cifar100 \
	model=resnet18 \
	output_model_path=resnet18.pt \
	trainer.num_clients=100 \
	trainer.participation_rate=0.05 \
	server=FedAdam_PPRL \
	trainer.encrypted=True \
	server.tau=0.00001 \
	trainer.enc_level=3 \
	split.alpha=0.3 \
	batch_size=50 \
	trainer.global_lr=0.01 \
	wandb=True
~~~

## Hyper-parameters
### Table. Hyperparameters for CIFAR-10/100 and Tiny-ImageNet. $\tau$ denotes the stabilization parameter; $t_1$ and $t_2$ denote moment thresholds; $T$, $K$, $\eta$, and $\eta_l$ denote the total number of rounds, local steps, server learning rate, and client learning rate, respectively.

| Hyperparameter | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|----------------|-----------|-------------|----------------|
| $\tau$         | $10^{-5}$ | $10^{-5}$  | $10^{-5}$      |
| $t_1$          | 110       | 100         | 110            |
| $t_2$          | 80        | 90          | 70             |
| $T$            | 1000      | 1000        | 1000           |
| $K$            | 5         | 5           | 5              |
| $\eta$         | 0.01      | 0.01        | 0.01           |
| $\eta_l$       | 0.1       | 0.1         | 0.1            |
| Batch size     | 50        | 50          | 50             |