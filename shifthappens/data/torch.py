"""Wrappers for PyTorch datasets such that they can be used as datasets
for the benchmark."""
import collections
import typing
from typing import Sequence

import torch
import torch.utils.data as ch_data

from .base import Dataset
from .base import IndexedDataset


def _convert_torch_value(value):
    if isinstance(value, (tuple, list)):
        converted_values = []
        for v in value:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            converted_values.append(v)
        return type(value)(converted_values)
    else:
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        return value


class TorchDataset(Dataset):
    """Wraps a torch iterable dataset
    (i.e. :py:class:`torch.utils.data.IterableDataset`)."""

    def __init__(self, torch_dataset: ch_data.IterableDataset):
        self.torch_dataset = torch_dataset

    def __iter__(self):
        self.torch_iter = iter(self.torch_dataset)
        return self

    def __next__(self):
        value = next(self.torch_iter)
        return _convert_torch_value(value)


class IndexedTorchDataset(IndexedDataset):
    """Wraps a torch map-style dataset
    (i.e. :py:class:`torch.utils.data.Dataset`)."""

    def __init__(self, torch_dataset: ch_data.Dataset):
        self.torch_dataset = torch_dataset

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, item):
        value = self.torch_dataset[item]
        return _convert_torch_value(value)


class ImagesOnlyTorchDataset(ch_data.Dataset):
    """Wraps a torch map-style dataset returning images and labels such that
    only the images are returned."""

    def __init__(self, dataset: ch_data.Dataset):
        assert hasattr(dataset, "__getitem__") and hasattr(
            dataset, "__len__"
        ), "Dataset must be map-style, i.e. implement a __len__ and __getitem__ method"
        self.dataset: Sequence = typing.cast(collections.Sequence, dataset)

    def __getitem__(self, index):
        return self.dataset[index][0]

    def __len__(self):
        return len(self.dataset)
