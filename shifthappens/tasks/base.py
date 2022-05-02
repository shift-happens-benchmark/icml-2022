"""Base definition of a task in the shift happens benchmark.

Fully defined tasks should subclass the :py:class:`Task` abstract base class, and implement all
mixins based on the required model outputs to evaluate the task, also part of this module.

Implementing a new task consists of the following steps:

1. Subclass the :py:class:`Task` class and implement its abstract methods to specify the task
   setup and evaluation scheme
2. Implement any number of mixins specified in this module. You just need to include
   the mixin in the class definition, e.g. ``class MyTask(Task, ConfidenceTaskMixin)``,
   and do not need to implement additional methods. My specifying the mixin, it will be
   assured that your task gets the correct model outputs.
   See the individual mixin classes, or the :py:class:`shifthappens.models.models.base.ModelResult`
   class for further details.
3. Register your class to the benchmark using the :py:func`shifthappens.benchmark.register_task`
   decorator, along with a name and data path for your benchmark. 
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
    """Task base class.

    Override the :py:meth:`setup`, :py:meth:`_prepare_dataloader` and :py:meth:`_evaluate` methods to define
    your task. Also make sure to add in additional mixins from :py:mod:`shifthappens.tasks.base` to specify
    which models your task is compatible to (e.g., specify that your task needs labels, or confidences from
    a model).

    To include the task in the benchmark, use the :py:func:`shifthappens.benchmark.register_task` decorator.
    """

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
        """Prepare a dataloader for just the images (i.e. no labels, etc.) which will be passed to the model
        before the actual evaluation. This allows models to, e.g., run unsupervised domain adaptation techniques.

        If intended, the implementation of this function should call the :py:func:`shifthappens.models.model.Model.prepare`
        function and pass (parts) of the data through a data loader. The model could potentially use this data for
        test-time adaptation, calibration, or orther purposes.

        Note that this function could also be used to create domain shift for such adaptation methods, by passing
        a different dataloader in this prepare function than used during :py:meth:`evaluate`.
        """
        raise NotImplementedError()

    @abstractmethod
    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluate the task and return a dictionary with the calculated metrics.

        model: The passed model implementents a ``predict`` function returning an iterator
        over :py:meth:`shifthappens.models.base.ModelResult`. Each result contains predictions such as
        the class labels assigned to the images, confidences, etc., based on which mixins were
        implemented by this task to request these prediction outputs.

        This function should return a :py:class:`shifthappens.tasks.task_result.TaskResult` which can
        contain an arbitrary dictionary of metrics, along with a specifiction of which of these
        metrics are main results/summary metrics for the task.
        """
        raise NotImplementedError()


class LabelTaskMixin:
    """Indicates that the task requires the model to return the predicted labels.

    Tasks implementing this mixin will be provided with the ``class_labels`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class ConfidenceTaskMixin:
    """Indicates that the task requires the model to return the confidence scores.

    Tasks implementing this mixin will be provided with the ``confidences`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class UncertaintyTaskMixin:
    """Indicates that the task requires the model to return the uncertainty scores.

    Tasks implementing this mixin will be provided with the ``uncertainties`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class OODScoreTaskMixin:
    """Indicates that the task requires the model to return the OOD scores.

    Tasks implementing this mixin will be provided with the ``ood_scores`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass


class FeaturesTaskMixin:
    """Indicates that the task requires the model to return the raw features.

    Tasks implementing this mixin will be provided with the ``features`` attribute in the
    :py:class:`shifthappens.models.base.ModelResult` returned during evaluation.
    """

    pass
