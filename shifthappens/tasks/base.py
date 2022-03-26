"""Base definition of a class in the shift-happens benchmark."""

from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import shifthappens.models.base as sh_models

# TODO implement flags for the task
# - dataloaders = {'stream', 'shuffle'}

@dataclasses.dataclass
class SpecifyDataLoaderMixin:

    shuffle_data: bool = True

    def _prepare(self):
        if shuffle_data:
            return ...
        else:
            return ...


class ShuffleEvalTask(SpecifyDataLoaderMixin, Task):

    blubb_my_arg: int = 42

    _config = TaskConfig(
        shuffle_data = [True, False],
        blubb = [0, 1, 2]
    )

    def setup(self):
        if self.blubb_my_arg == 73:
            print("hallo")

@dataclasses.dataclass
class Task(ABC):
    """Task base class."""

    def __post_init__(self):
        self.setup()

    @classmethod
    def iterate_instances(cls) -> Task:
        """Iterate over all possible task configurations."""
        configs = product_dict(cls._config)
        for config in configs:
            yield cls(config)

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError

    def evaluate(self, model: sh_models.Model) -> Optional[Dict[str, float]]:
        """"Validates that the model is compatible with the task and then evaluates the model's
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

        model.prepare(self._get_dataset())
        return self._evaluate(model)

    @abstractmethod
    def _prepare(self, model: sh_models.Model) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self, model: sh_models.Model) -> Dict[str, float]:
        """Implement this function to evaluate the task and return a dictionary with the
        calculated metrics."""
        raise NotImplementedError()


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
