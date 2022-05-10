"""Base definition of a task in the shift happens benchmark.

Fully defined tasks should subclass the :py:class:`Task` abstract base class, and implement all
mixins based on the required model outputs to evaluate the task, also part of this module.

Implementing a new task consists of the following steps:

1. Subclass the :py:class:`Task` class and implement its abstract methods to specify the task
   setup and evaluation scheme
2. Implement any number of mixins specified in :py:mod:`shifthappens.tasks.mixins`. You just need to include
   the mixin in the class definition, e.g. ``class MyTask(Task, ConfidenceTaskMixin)``,
   and do not need to implement additional methods. By specifying the mixin, it will be
   assured that your task gets the correct model outputs.
   See the individual mixin classes, or the :py:class:`ModelResult <shifthappens.models.base.ModelResult>`
   class for further details.
3. Register your class to the benchmark using the :py:func:`register_task <shifthappens.benchmark.register_task>`
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
import shifthappens.models.mixins as shm_mixins
import shifthappens.task_data.task_metadata
import shifthappens.tasks.mixins as sht_mixins
import shifthappens.utils as sh_utils
from shifthappens.data.base import DataLoader
from shifthappens.tasks.task_result import TaskResult

#: A generic representing arbitrary types.
T = TypeVar("T")


def parameter(default: T, options: Tuple[T, ...], description: Optional[str] = None):
    """Register a task's parameter. Setting multiple options here allows automatically
    creating different flavours of the task. Use this field for storing values of a hyperparameter
    if you want to run task with different hyperparameter. For example, you can use this mechanism
    to create tasks with varying difficulties without creating multiple classes/tasks.

    Args:
        default: default value.
        options: allowed options.
        description: short description.

    Examples:
        >>> @dataclasses.dataclass
        >>> class CustomTask(Task):
        >>>     max_batch_size: Optional[int] = shifthappens.tasks.base.parameter(
        >>>     default=typing.cast(Optional[int], None),
        >>>     options=(32, 64, 128, None), #None corresponds to dataset-sized batch
        >>>     description="maximum size of batches fed to the model during evaluation",
        >>>     )
                ...
    """
    assert len(options) > 0
    return dataclasses.field(
        default=default,
        repr=True,
        metadata=dict(is_parameter=True, description=description, options=options),
    )


def variable(value: T):
    """Creates a non-parametric variable for a task, i.e. its value won't be passed to the constructor.
    Use it to store constants such as links to the data.

    Args:
        value: value of the constant.

    Examples:
        >>> @dataclasses.dataclass
        >>> class CustomTask(Task):
        >>>     constant: str = shifthappens.tasks.base.variable("your constant")
                ...
    """

    return dataclasses.field(
        default_factory=lambda: value,
        init=False,
        repr=False,
    )


def abstract_variable():
    """Marks a variable as abstract such that a child class needs to override it
    with a non-abstract variable. See :py:func:`variable` for the non-abstract
    counterpart.

    Examples:
        >>> @dataclasses.dataclass
        >>> class CustomTask(Task):
        >>>     constant: str = shifthappens.tasks.base.abstract_variable()
                ...
        >>> @dataclasses.dataclass
        >>> class InheritedTask(CustomTask):
        >>>     constant: str = shifthappens.tasks.base.variable("your constant")
                ...
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

    To include the task in the benchmark, use the :py:func:`register_task <shifthappens.benchmark.register_task>`
    decorator.

    Args:
        data_root: Folder where individual tasks can store their data.
            This field is initialized with the value passed to
            :py:meth:`shifthappens.benchmark.evaluate_model`.
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
        """Set the task up, i.e., download, load and prepare the dataset.

        Examples:
            >>> # imagenet_r example
            >>> @shifthappens.benchmark.register_task(
            >>> ...
            >>> )
            >>> @dataclasses.dataclass
            >>> class ImageNetR(Task):
            >>>     ...
            >>>     def setup(self):
            >>>         dataset_folder = os.path.join(self.data_root, "imagenet-r")
            >>>         if not os.path.exists(dataset_folder): # download data
            >>>             for file_name, url, md5 in self.resources:
            >>>                 sh_utils.download_and_extract_archive(
            >>>                     url, self.data_root, md5, file_name
            >>>                 )
            >>>         ...
        """
        pass

    def evaluate(self, model: sh_models.Model) -> Optional[TaskResult]:
        """Validates that the model is compatible with the task and then evaluates the model's
        performance using the :py:meth:`_evaluate` function of this class.

        Args:
            model: The model to evaluate.
                See :py:meth:`_evaluate` for more details.
        """
        if issubclass(type(self), sht_mixins.ConfidenceTaskMixin) and not issubclass(
            type(model), shm_mixins.ConfidenceModelMixin
        ):
            return None

        if issubclass(type(self), sht_mixins.FeaturesTaskMixin) and not issubclass(
            type(model), shm_mixins.FeaturesModelMixin
        ):
            return None

        if issubclass(type(self), sht_mixins.LabelTaskMixin) and not issubclass(
            type(model), shm_mixins.LabelModelMixin
        ):
            return None

        dataloader = self._prepare_dataloader()
        if dataloader is not None:
            model.prepare(dataloader)
        return self._evaluate(model)

    def _prepare_dataloader(self) -> Optional[DataLoader]:
        """Prepare a :py:class:`shifthappens.data.base.DataLoader` based on just the *unlabeled* images which will be passed to the model
        before the actual evaluation. This allows models to, e.g., run unsupervised domain adaptation techniques.
        This method can be used to give models access to the unlabeled data in case they want to run some
        kind of unsupervised adaptation mechanism such as re-calibration.

        Note that this function could also be used to introduce domain shifts for such adaptation methods, by creating
        a different dataloader in this prepare function than used during :py:meth:`evaluate`.

        By default no `DataLoader <shifthappens.data.base.DataLoader>` is returned, i.e., the models do not get access to the unlabeled data.

        Examples:
            >>> @dataclasses.dataclass
            >>> class CustomTask(Task):
            >>>     ...
            >>>     def _prepare_dataloader(self) -> DataLoader:
            >>>         ...
            >>>         return shifthappens.data.base.DataLoader(dataset, max_batch_size)
            >>>     ...
        """
        return None

    @abstractmethod
    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluate the task and return a dictionary with the calculated metrics.

        Args:
            model (shifthappens.models.base.Model): The passed model implements a ``predict`` function returning an iterator
                over :py:meth:`shifthappens.models.base.ModelResult`. Each result contains predictions such as
                the class labels assigned to the images, confidences, etc., based on which mixins were
                implemented by this task to request these prediction outputs.

        Returns:
            :py:class:`shifthappens.tasks.task_result.TaskResult`: The results of the task in the
            form of a :py:class:`shifthappens.tasks.task_result.TaskResult` containing an
            arbitrary dictionary of metrics, along with a specification of which of these
            metrics are main results/summary metrics for the task.

        Examples:
            >>> # imagenet_r example
            >>> @shifthappens.benchmark.register_task(
            >>> ...
            >>> )
            >>> @dataclasses.dataclass
            >>> class ImageNetR(Task):
            >>>     ...
            >>>         def _evaluate(self, model: shifthappens.models.base.Model) -> TaskResult:
            >>>             dataloader = self._prepare_dataloader()
            >>>             all_predicted_labels_list = []
            >>>             for predictions in model.predict(
            >>>                 dataloader, PredictionTargets(class_labels=True)
            >>>             ):
            >>>                 all_predicted_labels_list.append(predictions.class_labels)
            >>>             all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)
            >>>
            >>>             accuracy = all_predicted_labels == np.array(self.ch_dataset.targets)
            >>>
            >>>             return TaskResult(
            >>>                 accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
            >>>             )
            >>>         ...
        """
        raise NotImplementedError()
