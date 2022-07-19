"""Base classes and helper functions for adding models to the benchmark.

To add a new model, implement a new wrapper class inheriting from
:py:class:`shifthappens.models.base.Model`, and from any of the Mixins defined
in :py:mod:`shifthappens.models.mixins`.

Model results should be converted to :py:class:`numpy.ndarray` objects, and
packed into an :py:class:`shifthappens.models.base.ModelResult` instance.
"""

import abc
import dataclasses
from typing import Iterator
import shifthappens.config

import numpy as np
from tqdm import tqdm

from shifthappens.data import imagenet as sh_imagenet
from shifthappens.data.base import DataLoader
from shifthappens.models import mixins


class ModelResult:
    """Emissions of a model after processing a batch of data.

    Each model needs to return class labels that are compatible with
    the ILSRC2012 labels. We use the same convention used by PyTorch
    regarding the ordering of labels.

    Args:
        class_labels: ``(N, k)``, top-k predictions for
            each sample in the batch. Choice of ``k`` can be selected by
            the user, and potentially influences the type of accuracy
            based benchmarks that the model can be run on. For standard
            ImageNet, ImageNet-C evaluation, choose at least ``k=5``.
        confidences: ``(N, 1000)``, confidences for each class.
            Standard PyTorch ImageNet class label order is expected for
            this array. Scores can be in the range ``-inf`` to ``inf``.
        uncertainties: ``(N, 1000)``, uncertainties for the
            different class predictions. Different from the ``confidences``,
            this is a measure of certainty of the given ``confidences`` and
            common e.g. in Bayesian Deep neural networks.
        ood_scores: ``(N,)``, score for interpreting the sample
            as an out-of-distribution class, in the range ``-inf`` to ``inf``.
        features: ``(N, d)``, where ``d`` can be arbitrary, feature
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
    """Contains boolean flags of which type of targets model is predicting. Note
    that at least one flag should be set as ``True`` and model should be inherited
    from corresponding ModelMixin.

    Args:
        class_labels: Set to ``True`` if model returns predicted labels.
        confidences: Set to ``True`` if model returns confidences.
        uncertainties: Set to ``True`` if model returns uncertainties.
        ood_scores: Set to ``True`` if model returns ood scores.
        features: Set to ``True`` if model returns features.
    """

    class_labels: bool = False
    confidences: bool = False
    uncertainties: bool = False
    ood_scores: bool = False
    features: bool = False

    def __post_init__(self):
        assert any(
            getattr(self, field.name) for field in dataclasses.fields(self)
        ), "At least one prediction target must be set."


class Model(abc.ABC):
    """Model base class.

    Override the :py:meth:`_predict` method to define predictions type of your specific model.
    If your model uses unsupervised adaptation mechanisms override :py:meth:`prepare`
    as well.

    Also make sure that your model inherits from the mixins from :py:mod:`shifthappens.models.mixins`
    corresponding to your model predictions type (e.g., :py:class:`LabelModelMixin <shifthappens.models.mixins.LabelModelMixin>` for labels
    or :py:class:`ConfidenceModelMixin <shifthappens.models.mixins.ConfidenceModelMixin>` for confidences).

    """

    def __init__(self):
        self._imagenet_validation_result = None
        self.verbose = False

    @property
    def imagenet_validation_result(self):
        """Access the model's predictions/evaluation results on the ImageNet validation set.

        Returns:
            Model evaluation result on ImageNet validation set wrapped with ModelResult.
        """
        if self._imagenet_validation_result is None:
            self._get_imagenet_predictions()

        return self._imagenet_validation_result

    def prepare(self, dataloader: DataLoader):
        """If the model uses unsupervised adaptation mechanisms, it will run those.

        Args:
            dataloader: Dataloader producing batches of data.
        """
        pass

    def predict(
        self, input_dataloader: DataLoader, targets: PredictionTargets
    ) -> Iterator[ModelResult]:
        """Yield all the predictions of the model for all data samples contained
        in the dataloader

        Args:
            input_dataloader: Dataloader producing batches of data.
            targets: Indicates which kinds of targets should
                be predicted.

        Returns:
            Prediction results for the given batch. Depending on the target
            arguments this includes the predicted labels, class confidences,
            class uncertainties, ood scores, and image features, all as
            :py:class:`numpy.ndarray` objects.
        """

        if targets.class_labels:
            assert issubclass(type(self), mixins.LabelModelMixin)
        if targets.confidences:
            assert issubclass(type(self), mixins.ConfidenceModelMixin)
        if targets.uncertainties:
            assert issubclass(type(self), mixins.UncertaintyModelMixin)
        if targets.ood_scores:
            assert issubclass(type(self), mixins.OODScoreModelMixin)
        if targets.features:
            assert issubclass(type(self), mixins.FeaturesModelMixin)

        return self._predict(input_dataloader, targets)

    def _get_imagenet_predictions(self, rewrite=False):
        """
        Loads cached predictions on ImageNet validation set for the model or predicts
        on ImageNet validation set and caches the result whenever there is no cached
        predictions for the model or ``rewrite`` argument set to ``True``.

        Args:
            rewrite: ``True`` if models predictions need to be rewritten.
        """
        if sh_imagenet.is_cached(self) and not rewrite:
            self._imagenet_validation_result = ModelResult(
                **sh_imagenet.get_cached_predictions(self)
            )
        else:
            self._predict_imagenet_val()
            sh_imagenet.cache_predictions(self, self.imagenet_validation_result)

    def _predict_imagenet_val(self):
        """
        Evaluates model on ImageNet validation set and store all possible targets scores
        for the particular model.
        """
        try:
            max_batch_size = getattr(self, "max_batch_size")
            imagenet_val_dataloader = sh_imagenet.get_imagenet_validation_loader(
                max_batch_size=max_batch_size
            )
        except AttributeError:
            imagenet_val_dataloader = sh_imagenet.get_imagenet_validation_loader()

        score_types = {
            "class_labels": issubclass(type(self), mixins.LabelModelMixin),
            "confidences": issubclass(type(self), mixins.ConfidenceModelMixin),
            "ood_scores": issubclass(type(self), mixins.OODScoreModelMixin),
            "uncertainties": issubclass(type(self), mixins.UncertaintyModelMixin),
            "features": issubclass(type(self), mixins.FeaturesModelMixin),
        }
        targets = PredictionTargets(**score_types)
        predictions_dict = {
            prediction_type: []
            for prediction_type in [
                score_type for score_type in score_types if score_types[score_type]
            ]
        }

        if self.verbose:
            pred_loader = tqdm(self._predict(imagenet_val_dataloader, targets), desc='Predictions', total=int(len(imagenet_val_dataloader._dataset)/imagenet_val_dataloader.max_batch_size))
        else:
            pred_loader = self._predict(imagenet_val_dataloader, targets)

        for prediction in pred_loader:
            for prediction_type in predictions_dict:
                prediction_score = prediction.__getattribute__(prediction_type)
                predictions_dict[prediction_type] = sum(
                    [predictions_dict[prediction_type], [prediction_score]], []
                )
        for prediction_type in predictions_dict:
            predictions_dict[prediction_type] = np.concatenate(
                predictions_dict[prediction_type], 0
            )
        self._imagenet_validation_result = ModelResult(**predictions_dict)

    @abc.abstractmethod
    def _predict(
        self, input_dataloader: DataLoader, targets: PredictionTargets
    ) -> Iterator[ModelResult]:
        """Override this function for the specific model.

        Args:
            input_dataloader: Dataloader producing batches of data.
            targets: Indicates which kinds of targets should be predicted.

        Returns:
            Yields prediction results for all batches yielded by the dataloader.
            Depending on the target arguments the model results may include the
            predicted labels, class confidences, class uncertainties, ood scores,
            and image features, all as :py:class:`numpy.ndarray` objects.
        """
        raise NotImplementedError()
