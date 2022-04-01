import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

import shifthappens.data.base as base_data
import shifthappens.data.torch as torch_data


def test_indexed_torch_dataset():
    torch_ds = TensorDataset(torch.empty((100, 32, 32, 3)), torch.empty((100, 2)))
    ds = torch_data.IndexedTorchDataset(torch_ds)

    for i in range(3):
        values = ds[i]
        assert len(values) == 2
        for value in values:
            assert isinstance(value, np.ndarray)


def test_torch_dataset():
    class DummyIterableDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            self.item = 0
            return self

        def __next__(self):
            self.item += 1
            if self.item > 10:
                raise StopIteration
            return torch.empty((32, 32, 3)), torch.empty(1)

    torch_ds = DummyIterableDataset()
    ds = torch_data.TorchDataset(torch_ds)

    for values in ds:
        assert len(values) == 2
        for value in values:
            assert isinstance(value, np.ndarray)


def test_dataloader():
    torch_ds = TensorDataset(torch.empty((100, 32, 32, 3)), torch.empty((100, 2)))
    ds = torch_data.IndexedTorchDataset(torch_ds)

    dl = base_data.DataLoader(ds, max_batch_size=3)
    for values in dl.iterate(5):
        assert 0 < len(values) <= 3
        for value in values:
            assert len(value) == 2


def test_shuffle_data():
    data = np.arange(100)

    # check that arguments need to be passed as named arguments
    with pytest.raises(TypeError):
        base_data.shuffle_data(data)  # type: ignore

    with pytest.raises(TypeError):
        base_data.shuffle_data(data, 1)  # type: ignore

    # check that seeding works
    shuffle_1 = base_data.shuffle_data(data=data, seed=1)
    shuffle_2 = base_data.shuffle_data(data=data, seed=1)
    assert np.all(shuffle_1 == shuffle_2)
