import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [ 'get_numclasses']

def get_numclasses(args,trainset = None):
    if args.dataset.name in ['CIFAR10', "MNIST"]:
        num_classes=10
    elif args.dataset.name in ["CIFAR100"]:
        num_classes=100
    elif args.dataset.name in ["TinyImageNet"]:
        num_classes=200
    elif args.dataset.name in ["iNaturalist"]:
        num_classes=1203
    elif args.dataset.name in ["ImageNet"]:
        num_classes=1000
    elif args.dataset.name in ["LEAF_CELEBA"]:
        num_classes = 2
    elif args.dataset.name in ["LEAF_FEMNIST"]:
        num_classes = 62
    elif args.dataset.name in ["LEAF_SHAKESPEARE"]:
        num_classes=80
    else:
        assert False
        
    print("num of classes of ", args.dataset.name," is : ", num_classes)
    return num_classes

