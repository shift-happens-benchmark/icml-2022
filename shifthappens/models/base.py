"""Base classes and helper functions for adding models to the benchmark.

To add a new model, implement a new wrapper class inheriting from
``shifthappens.base.Model``, and from any of the Mixins defined
in this module.

Model results should be converted to numpy arrays, and packed into an
``shifthappens.base.ModelResult`` instance.
"""

import abc
import dataclasses

import numpy as np

from shifthappens.data.base import DataLoader


class ModelResult:
    """Emissions of a model after processing a batch of data.

    Each model needs to return class labels that are compatible with
    the ILSRC2012 labels. We use the same convention used by PyTorch
    regarding the ordering of labels.

    Args:
        class_labels: ``(N, k)`` array containing top-k predictions for
            each sample in the batch. Choice of ``k`` can be selected by
            the user, and potentially influences the type of accuracy
            based benchmarks that the model can be run on. For standard
            ImageNet, ImageNet-C evaluation, choose at least ``k=5``.
        confidences: optional, ``(N, 1000)`` confidences for each class.
            Standard PyTorch ImageNet class label order is expected for
            this array. Scores can be in the range ``-inf`` to ``inf``.
        uncertainties: optional, ``(N, 1000)``, uncertainties for the
            different class predictions. Different from the ``confidences``,
            this is a measure of certainty of the given ``confidences`` and
            common e.g. in Bayesian Deep neural networks.
        ood_scores: optional, ``(N,)``, score for interpreting the sample
            as an out-of-distribution class, in the range ``-inf`` to ``inf``.
        features: optional, ``(N, d)``, where ``d`` can be arbitrary, feature
            representation used to arrive at the given predictions.
    """

    __slots__ = [
        "class_labels",
        "confidences",
        "uncertainties",
        "ood_scores",
        "features",
    ]

    def __init__(
        self,
        class_labels: np.ndarray,
        confidences: np.ndarray = None,
        uncertainties: np.ndarray = None,
        ood_scores: np.ndarray = None,
        features: np.ndarray = None,
    ):
        self.class_labels = class_labels
        self.confidences = confidences
        self.uncertainties = uncertainties
        self.ood_scores = ood_scores
        self.features = features


@dataclasses.dataclass
class PredictionTargets:
    class_labels: bool = False
    logits: bool = False
    confidences: bool = False
    uncertainties: bool = False
    ood_scores: bool = False
    features: bool = False

    def __post_init__(self):
        assert any(
            getattr(self, field.name) for field in dataclasses.fields(self)
        ), "At least one prediction target must be set."


class Model(abc.ABC):
    """Model base class."""

    def prepare(self, dataloader: DataLoader):
        pass

    def predict(self, inputs: np.ndarray, targets: PredictionTargets) -> ModelResult:
        """
        Args:
            inputs (np.ndarray): Batch of images.
            targets (PredictionTargets): Indicates which kinds of targets should be predicted.

        Returns:
            Prediction results for the given batch. Depending in the target arguments this
            includes the predicted labels, class confidences, class uncertainties, ood scores,
            and image features, all as ``np.array``s.
        """

        if issubclass(type(self), LabelModelMixin):
            assert targets.logits
        if issubclass(type(self), ConfidenceModelMixin):
            assert targets.confidences
        if issubclass(type(self), UncertaintyModelMixin):
            assert targets.uncertainties
        if issubclass(type(self), OODScoreModelMixin):
            assert targets.ood_scores
        if issubclass(type(self), FeaturesModelMixin):
            assert targets.features

        return self._predict(inputs, targets)

    @abc.abstractmethod
    def _predict(self, inputs: np.ndarray, targets: PredictionTargets) -> ModelResult:
        """
        Override this function for the specific model.

        Args:
            inputs (np.ndarray): Batch of images.
            targets (PredictionTargets): Indicates which kinds of targets should be predicted.

        Returns:
            Prediction results for the given batch. Depending in the target arguments this
            includes the predicted labels, class confidences, class uncertainties, ood scores,
            and image features, all as ``np.array``s.
        """
        raise NotImplementedError()


class LabelModelMixin:
    """Inherit from this class if your model returns predicted labels."""

    pass


class ConfidenceModelMixin:
    """Inherit from this class if you model returns confidences."""

    pass


class UncertaintyModelMixin:
    """Inherit from this class if your model returns uncertainties."""

    pass


class OODScoreModelMixin:
    """Inherit from this class if your model returns ood scores."""

    pass


class FeaturesModelMixin:
    """Inherit from this class if your model returns features."""

    pass
