"""Base definition of a task in the shift happens benchmark."""

import dataclasses
from abc import ABC
from abc import abstractmethod
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar

import shifthappens.task_data.task_metadata
import shifthappens.models.base as sh_models
import shifthappens.utils as sh_utils
from shifthappens.data.base import DataLoader
from shifthappens.tasks.task_result import TaskResult

T = TypeVar("T")


def parameter(default: T, options: Tuple[T, ...], description: Optional[str] = None):
    """Register a task's parameter. Setting multiple options here allows automatically
    creating different flavours of the test.

    Args:
        default (T): default value
        options (Tuple(T)): allowed options
        description (str): short description
    """
    assert len(options) > 0
    return dataclasses.field(
        default=default,
        repr=True,
        metadata=dict(is_parameter=True, description=description, options=options),
    )


def variable(value: T):
    """Creates a non-parametric variable for a task.

    Args:
        value (T): value of the constant
    """
    return dataclasses.field(
        default_factory=lambda: value,
        init=False,
        repr=False,
    )


def abstract_variable():
    """Marks a variable as abstract such that a child class needs to override it
    with a non-abstract variable.
    """

    return dataclasses.field(
        default=None,
        init=False,
        metadata=dict(is_abstract_variable=True),
    )


@dataclasses.dataclass  # type: ignore
class Task(ABC):
    """Task base class."""

    data_root: str

    def __post_init__(self):
        # check for abstract variables
        for field in dataclasses.fields(self):
            if field.metadata is not None and field.metadata.get(
                "is_abstract_variable", False
            ):
                raise TypeError(
                    f"Cannot initialize class {type(self)} since field {field.name} is "
                    f"marked as an abstract (i.e., not available via __init__) variable and "
                    f"must be overridden with an actual value."
                )

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
    def iterate_flavours(cls, **kwargs) -> Iterator["Task"]:
        """Iterate over all possible task configurations, i.e. different settings of parameter fields."""
        assert hasattr(
            cls, shifthappens.task_data.task_metadata._TASK_METADATA_FIELD
        ), "Flavors can only be iterated once a task has been registered"
        metadata = getattr(
            cls, shifthappens.task_data.task_metadata._TASK_METADATA_FIELD
        )
        parameter_options = cls.__get_all_parameter_options()
        for config in sh_utils.dict_product(parameter_options):
            flavored_task = cls(**config, **kwargs)
            name_sfx = ""
            for _, v in config.items():
                name_sfx += f"_{repr(v)}"
            flavored_metadata = shifthappens.task_data.task_metadata.TaskMetadata(
                name=metadata.name + name_sfx,
                relative_data_folder=metadata.relative_data_folder,
                standalone=metadata.standalone,
            )
            setattr(
                flavored_task,
                shifthappens.task_data.task_metadata._TASK_METADATA_FIELD,
                flavored_metadata,
            )
            yield flavored_task

    @abstractmethod
    def setup(self):
        """Set the task up, i.e., download, load and prepare the dataset."""
        pass

    def evaluate(self, model: sh_models.Model) -> Optional[TaskResult]:
        """Validates that the model is compatible with the task and then evaluates the model's
        performance using the _evaluate function of this class."""
        if issubclass(type(self), ConfidenceTaskMixin) and not issubclass(
            type(model), sh_models.ConfidenceModelMixin
        ):
            return None

        if issubclass(type(self), FeaturesTaskMixin) and not issubclass(
            type(model), sh_models.FeaturesModelMixin
        ):
            return None

        if issubclass(type(self), LabelTaskMixin) and not issubclass(
            type(model), sh_models.LabelModelMixin
        ):
            return None

        dataloader = self._prepare_dataloader()
        if dataloader is not None:
            model.prepare(dataloader)
        return self._evaluate(model)

    @abstractmethod
    def _prepare_dataloader(self) -> Optional[DataLoader]:
        """Prepares a dataloader for just the images (i.e. no labels, etc.) which will be passed to the model
        before the actual evaluation. This allows models to, e.g., run unsupervised domain adaptation techniques."""
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
