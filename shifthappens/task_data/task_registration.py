from dataclasses import dataclass
from typing import Type

from shifthappens.task_data.task_metadata import TaskMetadata
from shifthappens.tasks.base import Task


@dataclass
class TaskRegistration:
    """Class for storing a task's registration for the benchmark."""

    cls: Type[Task]
    metadata: TaskMetadata

    def __hash__(self):
        return hash(self.cls)
