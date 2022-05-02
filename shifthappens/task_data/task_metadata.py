from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class TaskMetadata:
    """Class for storing a task's metadata required by the task registration mechanism."""

    name: str
    relative_data_folder: str
    standalone: bool = True


_TASK_METADATA_FIELD = "__task_metadata__"
