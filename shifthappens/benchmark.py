from dataclasses import dataclass
from tasks.base import Task
from typing import Tuple

__all__ = ["registered_tasks", "run", "register_task"]


@dataclass
class TaskRegistration:
    task_cls: Task
    standalone_task: bool = True


__registered_tasks: TaskRegistration = []


@property
def registered_tasks() -> Tuple[TaskRegistration]:
    """All tasks currently registered as part of the benchmark."""
    return tuple(__registered_tasks)


def register_task(cls, *, standalone_task: bool = True):
    """Register as task as part of the benchmark.
    
    Args:
    standalone_task (bool): Is this task meaningful as a stand-alone task or 
        will this only be relevant as a part of a collection of tasks?
    """

    assert issubclass(cls, Task)
    if cls in [t.task_cls for t in registered_tasks]:
        return
    __registered_tasks.append(TaskRegistration(cls, standalone_task=standalone_task))
    return cls


def _is_compatible(tasks, model):
    # TODO
    # based on defined Mixins, check if model and task
    # are compatible
    return True

class ScoreCard():

    def summary():
        """return dataframe with
        
        index         |   columns
        --------------------------
        task
        """
        pass

    def per_task(self, task):
        pass



def run(model):
    """
    Runs all tasks of the benchmarks for this supplied model.
    
    Args:
    model (Model): 

    Returns:

    """

    results = ScoreCardEntry()

    for task_registration in registered_tasks:
        task_cls = task_registration.task_cls
        if not _is_compatible(task, model):
            continue
        for task in task_cls.iterate_instances():
            results.add(task.run())
