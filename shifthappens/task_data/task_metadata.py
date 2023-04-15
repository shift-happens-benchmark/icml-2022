"""Class for storing a task's metadata."""

from dataclasses import dataclass
import json


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

    def serialize_task_metadata(self) -> str:
        """
        Serialize TaskMetadata object into json string.
        """
        metadata_dict = {
            'name': self.name,
            'relative_data_folder': self.relative_data_folder,
            'standalone': self.standalone
        }
        return json.dumps(metadata_dict)

    @staticmethod
    def deserialize_task_metadata(metadata_str: str):
        """
        Deserialize valid json string into TaskMetadata object.
        """
        metadata_dict = json.loads(metadata_str)
        metadata = TaskMetadata(
            name=metadata_dict['name'],
            relative_data_folder= metadata_dict['relative_data_folder'],
            standalone=metadata_dict['standalone']
        )
        return metadata

_TASK_METADATA_FIELD = "__task_metadata__"
