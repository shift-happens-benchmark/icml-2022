import torch
from torch.utils.data import TensorDataset

import shifthappens.data.base as base_data
import shifthappens.data.torch as torch_data
from shifthappens.models import torchvision as torchvison_models
from shifthappens.models.base import PredictionTargets


def _create_mock_dataloader():
    torch_ds = TensorDataset(torch.empty((20, 32, 32, 3)), torch.empty((20, 2)))
    ds = torch_data.ImagesOnlyTorchDataset(torch_ds)
    ds = torch_data.IndexedTorchDataset(ds)
    dl = base_data.DataLoader(ds, max_batch_size=3)

    return dl


def test_torchvision_rn18():
    model = torchvison_models.resnet18(2)
    dl = _create_mock_dataloader()

    for result in model.predict(
        dl, PredictionTargets(class_labels=True, features=True, confidences=True)
    ):
        assert result.class_labels is not None
        assert result.features is not None
        assert result.confidences is not None

        assert (
            len(result.class_labels)
            == len(result.features)
            == len(result.confidences)
            == 2
        )


def test_torchvision_vgg16():
    model = torchvison_models.vgg16(2)
    dl = _create_mock_dataloader()

    for result in model.predict(
        dl, PredictionTargets(class_labels=True, features=True, confidences=True)
    ):
        assert result.class_labels is not None
        assert result.features is not None
        assert result.confidences is not None

        assert (
            len(result.class_labels)
            == len(result.features)
            == len(result.confidences)
            == 2
        )
