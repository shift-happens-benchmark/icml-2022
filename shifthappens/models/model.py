"""Base classes and helper functions for adding models to the benchmark.

To add a new model, implement a new wrapper class inheriting from
``shifthappens.models.Model``, and from any of the Mixins defined
in this module.

Model results should be converted to numpy arrays, and packed into an
``shifthappens.models.ModelResult`` instance.
"""

import abc

import numpy as np


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
        "class_labels"
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
        self.uncertainty = uncertainties
        self.ood_scores = ood_scores
        self.features = features


class Model(abc.ABC):
    """Model base class."""

    @abc.abstractmethod
    def predict(self, inputs: np.ndarray) -> ModelResult:
        """
        Args:
            inputs: Batch of images

        Returns:
            Prediction results for the given batch, including predicted labels,
            and optionally, class confidences, class uncertainties, ood scores,
            and image features, all as ``np.array``s.
        """
        raise NotImplementedError()
