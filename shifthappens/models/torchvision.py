"""Model baselines from torchvision."""

from typing import Iterator
from typing import List

import numpy as np
import torch
import torchvision
from surgeon_pytorch import Inspect
from torch import nn
from torchvision.transforms import functional as tv_functional

import shifthappens.models.base as sh_models
import shifthappens.models.mixins as sh_mixins
from shifthappens.data.base import DataLoader


class __TorchvisionPreProcessingMixin:
    """Performs the default preprocessing for torchvision ImageNet classifiers."""

    def _pre_process(self, batch: List[np.ndarray], device: str) -> torch.Tensor:
        inputs = []

        for item in batch:
            assert isinstance(item, np.ndarray)
            item_t = torch.tensor(item.transpose((2, 0, 1)))
            item_t = tv_functional.resize(item_t, 256)
            item_t = tv_functional.center_crop(item_t, 224)
            item_t = tv_functional.normalize(
                item_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            inputs.append(item_t)

        inputs_t = torch.stack(inputs, 0)
        inputs_t = inputs_t.to(device)
        return inputs_t


class __TorchvisionModel(
    sh_models.Model,
    __TorchvisionPreProcessingMixin,
    sh_mixins.LabelModelMixin,
    sh_mixins.ConfidenceModelMixin,
    sh_mixins.FeaturesModelMixin,
    sh_mixins.OODScoreModelMixin,
):
    """Wraps a torchvision model.

    Args:
        model: Pretrained torchvision model.
        max_batch_size: How many samples allowed per batch to load.
        feature_layer: Name layer which outputs logits.
        device: Selected device to run the model on.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_layer: str,
        max_batch_size: int,
        device: str = "cpu",
    ):
        assert not issubclass(
            type(model), torch.nn.DataParallel
        ), "Parallel models are not yet supported"
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = device
        self.hooked_model = Inspect(self.model, layer=feature_layer)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict(
        self, input_dataloader: DataLoader, targets: sh_models.PredictionTargets
    ) -> Iterator[sh_models.ModelResult]:
        for batch in input_dataloader.iterate(self.max_batch_size):
            # pre-process batch
            inputs = self._pre_process(batch, self.device)
            logits, features = self.hooked_model(inputs)
            features = features.view(len(features), -1)
            logits, features = logits.cpu(), features.cpu()
            probabilities = torch.softmax(logits, -1)
            max_confidences, predictions = probabilities.max(1)

            yield sh_models.ModelResult(
                class_labels=predictions.numpy(),
                confidences=probabilities.numpy(),
                ood_scores=max_confidences.numpy(),
                features=features.numpy(),
            )


class ResNet18(__TorchvisionModel):
    def __init__(
        self,
        max_batch_size: int,
        model: nn.Module = torchvision.models.resnet18(pretrained=True),
        feature_layer: str = "avgpool",
        device: str = "cpu",
    ):
        super().__init__(model, feature_layer, max_batch_size, device)


def resnet18(max_batch_size: int = 16, device: str = "cpu"):
    """
    Load a ResNet18 network trained on the ImageNet 2012 train set from torchvision.
    See :py:func:`torchvision.models.resnet18` for details.

    Args:
        max_batch_size: How many samples allowed per batch to load.
        device: Selected device to run the model on.
    """
    return __TorchvisionModel(
        torchvision.models.resnet18(pretrained=True),
        "avgpool",
        max_batch_size=max_batch_size,
        device=device,
    )


def resnet50(max_batch_size: int = 16, device: str = "cpu"):
    """Load a ResNet50 network trained on the ImageNet 2012 train set from torchvision.
    See :py:func:`torchvision.models.resnet50` for details.

    Args:
        max_batch_size: How many samples allowed per batch to load.
        device: Selected device to run the model on.
    """
    return __TorchvisionModel(
        torchvision.models.resnet50(pretrained=True),
        "avgpool",
        max_batch_size=max_batch_size,
        device=device,
    )


def vgg16(max_batch_size: int = 16, device: str = "cpu"):
    """Load a VGG16 network trained on the ImageNet 2012 train set from torchvision.
    See :py:func:`torchvision.models.vgg16` for details.

    Args:
        max_batch_size: How many samples allowed per batch to load.
        device: Selected device to run the model on.
    """
    return __TorchvisionModel(
        torchvision.models.vgg16(pretrained=True),
        "avgpool",
        max_batch_size=max_batch_size,
        device=device,
    )
