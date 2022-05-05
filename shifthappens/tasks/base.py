"""Base classes and helper functions for adding tasks to the benchmark.

To add a new task, implement a new wrapper class inheriting
from :py:class:`shifthappens.tasks.base.Task`, and from any of the Mixins defined in this module.

Model results should be stored as a dictionary,
and packed into an :py:class:`shifthappens.tasks.task_result.TaskResult` instance.
"""

import dataclasses
from abc import ABC
from abc import abstractmethod
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar

import shifthappens.models.base as sh_models
import shifthappens.task_data.task_metadata
import shifthappens.utils as sh_utils
from shifthappens.data.base import DataLoader
from shifthappens.tasks.task_result import TaskResult

T = TypeVar("T")


def parameter(default: T, options: Tuple[T, ...], description: Optional[str] = None):
    """Register a task's parameter. Setting multiple options here allows automatically
    creating different flavours of the task.

    Args:
        default (T): default value
        options (Tuple(T)): allowed options
        description (str): short description

    Examples:
        >>> import dataclasses
        >>> from shifthappens.tasks.base import Task
        >>> @dataclasses.dataclass
        >>> class CustomTask(Task):
                max_batch_size: Optional[int] = parameter(
                    default=typing.cast(Optional[int], None),
                    options=(32, 64, 128, None), #None corresponds to dataset-sized batch
                    description="maximum size of batches fed to the model during evaluation",
                    )
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
        """Iterate over all possible task configurations,
        i.e. different settings of parameter fields. Parameters should be defined
        with :py:meth:`shifthappens.tasks.base.parameter`, where ``options`` argument
        corresponds to possible configurations of particular parameter."""
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
        """Validates that the model is compatible with the task and then evaluates
        the model's performance using the _evaluate function of this class.

        Args:
            model (shifthappens.models.base.Model): A model inherited from :py:class:`shifthappens.models.base.Model`.

        """
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
        """Prepares a dataloader for just the images (i.e. no labels, etc.)
        which will be passed to the model before the actual evaluation.
        This allows models to, e.g., run unsupervised domain adaptation techniques."""
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
