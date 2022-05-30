from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class TaskMetadata:
    """Class for storing a task's metadata required by the task registration mechanism.
    Arguments are passed by :py:meth:`shifthappens.benchmark.register_task` and are the same.

    Args:
        name: Name of the task (can contain spaces or special characters).
        relative_data_folder: Name of the folder in which the data for
            this dataset will be saved for this task relative to the root folder
            of the benchmark.
        standalone: Boolean which represents if this task meaningful as a
            standalone task or will this only be relevant as a part of a
            collection of tasks.
    """

    name: str
    relative_data_folder: str
    standalone: bool = True


_TASK_METADATA_FIELD = "__task_metadata__"
