from .base import Dataset, IndexedDataset
import torch


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
    """Wraps a torch iterable dataset (i.e. torch.utils.data.IterableDataset)."""
    def __init__(self, torch_dataset: torch.utils.data.IterableDataset):
        self.torch_dataset = torch_dataset

    def __iter__(self):
        self.torch_iter = iter(self.torch_dataset)
        return self

    def __next__(self):
        value = next(self.torch_iter)
        return _convert_torch_value(value)


class IndexedTorchDataset(IndexedDataset):
    """Wraps a torch map-style dataset (i.e. torch.utils.data.Dataset)."""
    def __init__(self, torch_dataset: torch.utils.data.Dataset):
        self.torch_dataset = torch_dataset

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, item):
        value = self.torch_dataset[item]
        return _convert_torch_value(value)
