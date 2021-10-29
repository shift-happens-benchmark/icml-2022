from typing import Dict
from typing import Optional
from abc import ABC, abstractmethod

from models import model


class ConfidenceTaskMixin:
    # inherit if your task requires the model to compute confidence scores
    pass


class FeaturesTaskMixin:
    # inherit if your task requires the model to compute features
    pass


class LabelTaskMixin:
    # inherit if your task requires the model to compute class labels
    pass


class Benchmark(ABC):
    def evaluate(self, model: model.Model) -> Optional[Dict[str, float]]:
        if issubclass(type(self), ConfidenceTaskMixin) and not issubclass(
            type(model), model.ConfidenceModelMixin
        ):
            return False

        if issubclass(type(self), FeaturesTaskMixin) and not issubclass(
            type(model), model.FeaturesModelMixin
        ):
            return False

        if issubclass(type(self), LabelTaskMixin) and not issubclass(
            type(model), model.LabelModelMixin
        ):
            return False

        return self._evaluate(model)

    @abstractmethod
    def _evaluate(model: model.Model) -> Dict[str, float]:
        """Implement this function to evaluate your task and return a dictionary with the calculated metrics."""
        pass
