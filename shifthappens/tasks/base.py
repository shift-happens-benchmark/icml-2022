"""Base definition of a class in the shift-happens benchmark."""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import shifthappens.models.base as sh_models


class Task(ABC):
    """Task base class."""

    def evaluate(self, model: sh_models.Model) -> Optional[Dict[str, float]]:
        """Validates that the model is compatible with the task and then evaluates the model's
        performance using the _evaluate function of this class."""
        if issubclass(type(self), ConfidenceTaskMixin) and not issubclass(
            type(model), model.ConfidenceModelMixin
        ):
            return None

        if issubclass(type(self), FeaturesTaskMixin) and not issubclass(
            type(model), model.FeaturesModelMixin
        ):
            return None

        if issubclass(type(self), LabelTaskMixin) and not issubclass(
            type(model), model.LabelModelMixin
        ):
            return None

        return self._evaluate(model)

    @abstractmethod
    def _evaluate(self, model: sh_models.Model) -> Dict[str, float]:
        """Implement this function to evaluate the task and return a dictionary with the
        calculated metrics."""
        pass


class LabelTaskMixin:
    """Indicates that the task requires the model to return the predicted labels."""

    pass


class ConfidenceTaskMixin:
    """Indicates that the task requires the model to return the confidence scores."""

    pass


class UncertaintyTaskMixin:
    """Indicates that the task requires the model to return the uncertainty scores."""

    pass


class OODScoreTaskMixin:
    """Indicates that the task requires the model to return the OOD scores."""

    pass


class FeaturesTaskMixin:
    """Indicates that the task requires the model to return the raw features."""

    pass
