"""Torchvision baselines."""
from typing import Iterator
from typing import List

import numpy as np
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as tv_functional

import shifthappens.models.base as sh_models
from shifthappens.data.base import DataLoader


class __TorchModel(
    sh_models.Model, sh_models.LabelModelMixin, sh_models.ConfidenceModelMixin, sh_models.OODScoreModelMixin
):
    """Wraps a torchvision model."""

    def __init__(self, model: nn.Module, max_batch_size: int, device: str = "cpu"):
        self.model = model
        self.max_batch_size = max_batch_size
        self.device = device

    def _pre_process(self, batch: List[np.ndarray]) -> torch.Tensor:
        inputs = []
        for item in batch:
            item_t = torch.tensor(item.transpose((2, 0, 1)))
            item_t = tv_functional.resize(item_t, 256)
            item_t = tv_functional.center_crop(item_t, 224)
            inputs.append(item_t)

        inputs_t = torch.stack(inputs, 0)
        inputs_t = inputs_t.to(self.device)
        return inputs_t

    @torch.no_grad()
    def _predict(
        self, input_dataloader: DataLoader, targets: sh_models.PredictionTargets
    ) -> Iterator[sh_models.ModelResult]:
        for batch in input_dataloader.iterate(self.max_batch_size):
            # pre-process batch
            inputs = self._pre_process(batch)
            logits = self.model(inputs).cpu()
            probabilities = torch.softmax(logits, -1)
            max_confidences, predictions = probabilities.max(-1)

            yield sh_models.ModelResult(
                class_labels=predictions.numpy(), confidences=probabilities.numpy(), ood_scores=-max_confidences.numpy()
            )


def resnet18(max_batch_size: int = 16, device: str = "cpu"):
    return __TorchModel(
        torchvision.models.resnet18(pretrained=True),
        max_batch_size=max_batch_size,
        device=device,
    )
