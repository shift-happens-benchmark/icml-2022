"""Base definition of a task in the shift-happens benchmark."""

from abc import ABC
from abc import abstractmethod
import dataclasses
from typing import Dict, TypeVar, Tuple, Iterator
from typing import Optional

import shifthappens.utils as sh_utils
import shifthappens.models.base as sh_models
from shifthappens.data.base import DataLoader
from shifthappens.tasks.task_result import TaskResult

"""
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
        shuffle_data=[True, False],
        blubb=[0, 1, 2]
    )

    def setup(self):
        if self.blubb_my_arg == 73:
            print("hallo")
"""

T = TypeVar('T')


def parameter(default: T, options: Tuple[T, ...], description: str = None):
    """Register a task's parameter. Setting multiple options here allows automatically
    creating different flavours of the test.

    Args:
        default (T): default value
        options (Tuple(T)): allowed options
        description (str): short description
    """
    assert len(options) > 0
    return dataclasses.field(
        default=default, repr=True,
        metadata=dict(is_parameter=True, description=description, options=options)
    )


@dataclasses.dataclass
class Task(ABC):
    """Task base class."""

    def __post_init__(self):
        self.setup()

    @classmethod
    def __get_all_parameters(cls):
        parameters = []
        fields = dataclasses.fields(cls)
        for field in fields:
            if field.metadata.get("is_parameter", False):
                parameters.append(field)
        return parameters

    @classmethod
    def __get_all_parameter_options(cls):
        parameters = cls.__get_all_parameters()
        parameter_options = {}
        for p in parameters:
            parameter_options[p.name] = p.metadata["options"]
        return parameter_options

    @classmethod
    def iterate_flavours(cls) -> Iterator["Task"]:
        """Iterate over all possible task configurations, i.e. different settings of parameter fields."""
        parameter_options = cls.__get_all_parameter_options()
        for config in sh_utils.dict_product(parameter_options):
            yield cls(**config)

    @abstractmethod
    def setup(self):
        pass

    def evaluate(self, model: sh_models.Model) -> Optional[TaskResult]:
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

        dataloader = self._prepare_data()
        model.prepare(dataloader)
        return self._evaluate(model)

    @abstractmethod
    def _prepare_data(self) -> DataLoader:
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self, model: sh_models.Model) -> TaskResult:
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
