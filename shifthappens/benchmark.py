from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric


__all__ = ["registered_tasks", "run", "register_task"]


@dataclass
class TaskRegistration:
    cls: Task
    name: str
    standalone: bool = True


__registered_tasks: List[TaskRegistration] = []


def get_registered_tasks() -> Tuple[TaskRegistration]:
    """All tasks currently registered as part of the benchmark."""
    return tuple(__registered_tasks)


def register_task(cls, *, name: str, standalone: bool = True):
    """Register as task as part of the benchmark.
    
    Args:
    standalone (bool): Is this task meaningful as a stand-alone task or 
        will this only be relevant as a part of a collection of tasks?
    """

    assert issubclass(cls, Task)
    if cls in [t.cls for t in registered_tasks]:
        return
    __registered_tasks.append(TaskRegistration(cls, name=name, standalone=standalone))
    return cls


def evaluate_model(model) -> Dict[TaskRegistration, Optional[TaskResult]]:
    """
    Runs all tasks of the benchmarks for the supplied model.
    
    Args:
    model (Model): 

    Returns:

    """

    results = dict()

    for task_registration in get_registered_tasks():
        if not task_registration.standalone:
            continue
        for task in task_registration.cls.iterate_flavours():
            results[task_registration] = task.evaluate(model)
    return results
