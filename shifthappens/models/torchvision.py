"""Torchvision baselines."""

import torch
from torch import nn as nn
import torchvision

import shifthappens.models.base as sh_models


class _ReturnFeatures(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._base = base

    def forward(self, inputs):
        return inputs, self._base(inputs)


class ResNet50(
    sh_models.Model,
    sh_models.LabelModelMixin,
    sh_models.ConfidenceModelMixin,
    sh_models.FeaturesModelMixin,
):
    """Reference implementation for a torchvision ResNet50 model."""

    def __init__(self):
        self._model = torchvision.models.resnet50(pretrained=True)
        self._model = _ReturnFeatures(self._model.fc)

    @torch.no_grad()
    def predict(self, inputs):
        inputs = torch.from_numpy(inputs)
        features, logits = self._model(inputs)
        predictions = torch.topk(logits, k=5, largest=True, sorted=True)

        return sh_models.ModelResult(
            class_labels=predictions.numpy(),
            confidences=logits.numpy(),
            features=features.numpy(),
        )
