from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import shifthappens.models.base as sh_models

class Task(ABC):
    """Task base class."""

    def evaluate(self, model: sh_models.Model) -> Optional[Dict[str, float]]:
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
    def _evaluate(model: sh_models.Model) -> Dict[str, float]:
        """Implement this function to evaluate your task and return a dictionary with the calculated metrics."""
        pass

class LabelTaskMixin:
    """Inherit from this class if your task returns predicted labels."""

    pass


class ConfidenceTaskMixin:
    """Inherit from this class if you task returns confidences."""

    pass


class UncertaintyTaskMixin:
    """Inherit from this class if your task returns uncertainties."""

    pass


class OODScoreTaskMixin:
    """Inherit from this class if your task returns ood scores."""

    pass


class FeaturesTaskMixin:
    """Inherit from this class if your task returns features."""

    pass



