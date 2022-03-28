import abc
from typing import Iterator
from typing import List
from typing import Union

import numpy as np


def shuffle_data(
    *, data: Union[List[np.ndarray], np.ndarray], seed: int
) -> Union[List[np.ndarray], np.ndarray]:
    """Randomly shuffles an numpy array/list of numpy arrays with a fixed random seed."""
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


class Dataset(abc.ABC):
    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


class IndexedDataset(Dataset):
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
    """

    def __init__(self, dataset, max_batch_size):
        self._dataset = dataset
        self.__max_batch_size = max_batch_size

    @property
    def max_batch_size(self):
        return self.__max_batch_size

    def iterate(self, batch_size) -> Iterator[List[np.ndarray]]:
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
