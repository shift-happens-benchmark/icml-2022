"""Base classes and helper functions for data handling (dataset, dataloader)."""

import abc
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import numpy as np


class Dataset(abc.ABC):
    """
    An abstract class representing an `iterable dataset <https://pytorch.org/docs/stable/data.html#iterable-style-datasets>`_.
    Your iterable datasets should be inherited from this class.
    """

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class IndexedDataset(Dataset):
    """
    A class representing a `map-style dataset <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_.
    Your map-style datasets should be inherited from this class.
    """

    @abc.abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        self.current_index = -1

        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self):
            raise StopIteration
        return self[self.current_index]


class DataLoader:
    """
    Interface b/w model and task, implements restrictions
    (e.g. max batch size) for models.

    Args:
            dataset: Dataset from which to load the data.
            max_batch_size: How many samples allowed per batch to load.
    """

    def __init__(self, dataset: Dataset, max_batch_size: Optional[int]):
        self._dataset = dataset
        self.__max_batch_size = max_batch_size

    @property
    def max_batch_size(self):
        return self.__max_batch_size

    def iterate(self, batch_size) -> Iterator[List[np.ndarray]]:
        if self.max_batch_size is not None:
            batch_size = min(batch_size, self.max_batch_size)

        dataset_exhausted = False
        ds_iter = iter(self._dataset)
        while not dataset_exhausted:
            batch = []
            try:
                for _ in range(batch_size):
                    batch.append(next(ds_iter))
            except StopIteration:
                dataset_exhausted = True

            yield batch


def shuffle_data(
    *, data: Union[List[np.ndarray], np.ndarray], seed: int
) -> Union[List[np.ndarray], np.ndarray]:
    """Randomly shuffles without replacement an :py:class:`numpy.ndarray`/list of
    :py:class:`numpy.ndarray` objects with a fixed random seed.

    Args:
            data: Data to shuffle.
            seed: Random seed.
    """
    undo_list = False
    if not isinstance(data, List):
        undo_list = True
        data = [
            data,
        ]
    assert np.all(
        [len(data[0]) == len(it) for it in data]
    ), "All data arrays must have the same length"
    rng = np.random.default_rng(seed=seed)
    rnd_indxs = rng.choice(len(data[0]), size=len(data[0]), replace=False)
    data = [it[rnd_indxs] for it in data]

    if undo_list:
        data = data[0]

    return data
