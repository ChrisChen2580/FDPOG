import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Subset
import numpy as np
from typing import Sequence

class IIDBatchSampler:
    def __init__(self, dataset, minibatch_size, iterations):
        self.length = len(dataset)
        self.minibatch_size = minibatch_size
        self.iterations = iterations

    def __iter__(self):
        for _ in range(self.iterations):

            indices = np.where(torch.rand(self.length) < (self.minibatch_size / self.length))[0]
            if indices.size > 0:
                yield indices

    def __len__(self):
        return self.iterations


def get_data_loaders_possion(minibatch_size, microbatch_size, iterations, drop_last=True):

    def minibatch_loader(dataset):
        sampler = IIDBatchSampler(dataset, minibatch_size, iterations)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
        )

    def microbatch_loader(minibatch):
        return DataLoader(
            minibatch,
            batch_size=microbatch_size,
            drop_last=drop_last,
        )

    return minibatch_loader, microbatch_loader