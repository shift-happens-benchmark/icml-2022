import os
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type

import shifthappens.utils as sh_utils
from shifthappens.models import Model
from shifthappens.tasks.base import Task
from shifthappens.tasks.task_result import TaskResult

__all__ = ["get_registered_tasks", "evaluate_model", "register_task"]


@dataclass
class TaskRegistration:
    cls: Type[Task]
    name: str
    relative_data_folder: str
    standalone: bool = True

    def __hash__(self):
        return hash(self.cls)


__registered_tasks: Set[TaskRegistration] = set()


def get_registered_tasks() -> Tuple[TaskRegistration]:
    """All tasks currently registered as part of the benchmark."""
    return tuple([x.cls for x in __registered_tasks])


def get_task_registrations() -> Tuple[TaskRegistration]:
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

    def _inner_register_task(cls: Type[Task]):
        assert issubclass(cls, Task)
        if cls in [t.cls for t in __registered_tasks]:
            return
        __registered_tasks.add(
            TaskRegistration(
                cls,
                name=name,
                relative_data_folder=relative_data_folder,
                standalone=standalone,
            )
        )
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
    model (Model):
    data_root (str): Folder where individual tasks can store their data.

    Returns:

    """

    results = dict()

    for task_registration in get_task_registrations():
        if not task_registration.standalone:
            continue
        for task in task_registration.cls.iterate_flavours(
            data_root=os.path.join(data_root, task_registration.relative_data_folder)
        ):
            results[task_registration] = task.evaluate(model)
    return results
