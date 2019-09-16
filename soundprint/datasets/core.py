from torch.utils.data import Dataset, Sampler
from torchvision import datasets
import torch
from abc import abstractmethod
from typing import List, Iterable
import numpy as np
import pandas as pd
import bisect


sex_to_label = {'M': False, 'F': True}
label_to_sex = {False: 'M', True: 'F'}




class PairDataset(Dataset):
    def __init__(self, dataset, pairs=None, labels=None, num_pairs=None):
        self.dataset = dataset
        self.pairs = pairs
        self.labels = labels
        if num_pairs is None:
            assert len(pairs) == len(labels)
        else:
            self.num_pairs = num_pairs

    def __len__(self):
        if self.num_pairs is None:
            return len(self.labels)
        else:
            return self.num_pairs

    def __getitem__(self, index):
        if self.pairs is not None:
            x = self.dataset[self.pairs[index][0]][0]
            y = self.dataset[self.pairs[index][0]][0]
            label = self.labels[index]
        else:
            index_1 = np.random.randint(len(self.dataset))
            index_2 = np.random.randint(len(self.dataset))
            x, x_label = self.dataset[index_1]
            y, y_label = self.dataset[index_2]
            label = x_label == y_label

        return (x, y), label


def collate_pairs(pairs):
    lefts = []
    rights = []
    labels = []
    for (x, y), label in pairs:
        lefts.append(x)
        rights.append(y)
        labels.append(label)

    x = torch.from_numpy(np.stack(lefts)).double()
    y = torch.from_numpy(np.stack(rights)).double()
    return (x, y), labels


class AudioDataset(Dataset):
    base_sampling_rate: int
    df: pd.DataFrame

    @property
    @abstractmethod
    def num_classes(self):
        raise NotImplementedError