import torch
from torchvision import datasets, transforms
import os
import torch.nn as nn
from typing import List, Dict
from collections import OrderedDict

import numpy as np

__all__ = ['DatasetSplit', 'DatasetSplitSubset', 'get_dataset']

create_dataset_log = False


class DatasetSplit(torch.utils.data.Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.class_dict = {}
        for idx in self.idxs:
            _, label = self.dataset[idx]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)
            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights



class DatasetSplitSubset(DatasetSplit):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, subset_classes=None):
        self.dataset = dataset

        self.subset_classes = subset_classes

        self.class_dict = {}
        self.indices = []

        for idx in idxs:
            _, label = self.dataset[int(idx)]
            if torch.is_tensor(label):
                label = str(label.item())
            else:
                label = str(label)

            if subset_classes is not None and int(label) not in subset_classes:
                continue

            self.indices.append(idx)

            if label in self.class_dict:
                self.class_dict[str(label)] += 1
            else:
                self.class_dict[str(label)] = 1


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image, label = self.dataset[self.indices[item]]
        return image, label
    
    @property
    def num_classes(self):
        return len(self.class_dict.keys())
    
    @property
    def class_ids(self):
        return self.class_dict.keys()
    
    def importance_weights(self, labels, pow=1):
        class_counts = np.array([self.class_dict[str(label.item())] for label in labels])
        weights = (1/class_counts)**pow
        weights /= weights.mean()
        return weights


def get_dataset(args, trainset, mode='iid'):
    dataset_name = args.dataset.name
    dataset_name_lc = dataset_name.lower()
    if 'leaf' in dataset_name_lc:
        return trainset.get_train_idxs()
    elif dataset_name_lc == 'shakespeare':
        return trainset.get_client_dic()


