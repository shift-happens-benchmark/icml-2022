import dataclasses
import os
from typing import Dict
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

import shifthappens.utils as sh_utils
from shifthappens.models import Model
from shifthappens.task_data import task_metadata
from shifthappens.task_data.task_registration import TaskRegistration
from shifthappens.tasks.base import Task
from shifthappens.tasks.task_result import TaskResult

__all__ = ["get_registered_tasks", "evaluate_model", "register_task"]
__registered_tasks: Set[TaskRegistration] = set()


def get_registered_tasks() -> Tuple[Type[Task], ...]:
    """All tasks currently registered as part of the benchmark."""
    return tuple([x.cls for x in __registered_tasks])


def get_task_registrations() -> Tuple[TaskRegistration, ...]:
    """Registrations for all tasks currently registered as part of the benchmark."""
    return tuple(__registered_tasks)


def register_task(*, name: str, relative_data_folder: str, standalone: bool = True):
    """Register as task as part of the benchmark.

    Args:
    name (str): Name of the task (can contain spaces or special characters).
    relative_data_folder (str): Name of the folder in which the data for this dataset will be saved for this task
        relative to the root folder of the benchmark.
    standalone (bool): Is this task meaningful as a stand-alone task or
        will this only be relevant as a part of a collection of tasks?
    """

    assert sh_utils.is_pathname_valid(
        relative_data_folder
    ), "relative_data_folder must only contain valid characters for a path"

    def _inner_register_task(cls: Type[Task], /):
        assert issubclass(cls, Task)
        # make sure the class was not registered before
        if cls in [t.cls for t in __registered_tasks]:
            return

        # check whether the task is marked as a dataclass
        # (i.e. defines its own _FIELDS attribute and does not use that of the base task)
        assert getattr(cls, getattr(dataclasses, "_FIELDS")) is not getattr(
            Task, getattr(dataclasses, "_FIELDS")
        ), "Tasks need to be dataclasses (i.e. add a @dataclasses.dataclass() decorator)"
        # check that the class did not define any fields the benchmark uses internally
        forbidden_fields = [task_metadata._TASK_METADATA_FIELD]
        for forbidden_field in forbidden_fields:
            assert not hasattr(
                cls, forbidden_field
            ), f"Tasks must not have an attribute called `{forbidden_field}`"

        # add metadata to class definition
        metadata = task_metadata.TaskMetadata(
            name=name,
            relative_data_folder=relative_data_folder,
            standalone=standalone,
        )
        setattr(cls, task_metadata._TASK_METADATA_FIELD, metadata)

        # finally register class
        registration = TaskRegistration(cls, metadata=metadata)
        __registered_tasks.add(registration)
        return cls

    return _inner_register_task


def unregister_task(cls: Type[Task]):
    """Unregisters a task by removing it from the task registry."""
    for cls_reg in __registered_tasks:
        if cls_reg.cls == cls:
            __registered_tasks.remove(cls_reg)
            return
    raise ValueError(f"Task `{cls}` is not registered.")


def evaluate_model(
    model: Model, data_root: str
) -> Dict[TaskRegistration, Optional[TaskResult]]:
    """
    Runs all tasks of the benchmarks for the supplied model.

    Args:
    model (Model): Model to evaluate.
    data_root (str): Folder where individual tasks can store their data.

    Returns (dict): Associates ``shifthappens.benchmark.TaskMetadata``s
        with the respective ``shifthappens.tasks.task_result.TaskResult``s.

    """

    results = dict()

    for task_registration in get_task_registrations():
        if not task_registration.metadata.standalone:
            continue
        for task in task_registration.cls.iterate_flavours(
            data_root=os.path.join(
                data_root, task_registration.metadata.relative_data_folder
            )
        ):
            task.setup()
            flavored_task_metadata = getattr(task, task_metadata._TASK_METADATA_FIELD)
            results[flavored_task_metadata] = task.evaluate(model)
    return results
