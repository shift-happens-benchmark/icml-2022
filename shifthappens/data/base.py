import abc

import torch.utils.data as torch_data

class Dataset(abc.ABC):

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def __next__(self):
        raise NotImplementedError


class DataLoader():
    """
    Interface b/w model and task, implements restrictions
    (e.g. max batch size) for models.
    """

    def __init__(self, dataset, max_batchsize):
        self._dataset = dataset
        self.__max_batchsize = max_batchsize

    @property
    def max_batchsize(self):
        return self.__max_batchsize

    def iterate(self, batch_size):

        if batch_size > self.max_batchsize:
            # TODO warning + ...
            batch_size = self.max_batchsize

        for ...:
            yield batch # List/tuple of np.array, len = batch_size
