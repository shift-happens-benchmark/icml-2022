"""CCC: Continuously Changing Corruptions

.. note::

    This task only implements the data reading portion of the dataset.
    In addition to this file, we submitted a file used to generate the
    data itself.
"""
import dataclasses

import numpy as np

import shifthappens.data.torch as sh_data_torch
from shifthappens import benchmark as sh_benchmark
from shifthappens.config import imagenet_validation_path
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import parameter
from shifthappens.tasks.base import Task
from shifthappens.tasks.ccc.ccc_utils import WalkLoader
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(name="CCC", relative_data_folder="ccc", standalone=True)
@dataclasses.dataclass
class CCC(Task):
    """
    The main task class for the CCC task.
    This task only implements the data reading portion of the dataset.
    """

    seed: int = parameter(
        default=43,
        options=(43, 44, 45),
        description="random seed used in the dataset building process",
    )
    frequency: int = parameter(
        default=5000,
        options=(5000, 10000, 20000),
        description="represents how many images are sampled from each subset",
    )
    base_amount: int = parameter(
        default=750000,
        options=(750000,),
        description="represents how large the base dataset is",
    )
    accuracy: int = parameter(
        default=20,
        options=(0, 20, 40),
        description="represents the baseline accuracy of walk",
    )
    subset_size: int = parameter(
        default=20000,
        options=(5000, 20000, 50000),
        description="represents the sample size of images sampled from ImageNet validation",
    )

    def setup(self):
        """Load and prepare the data."""

        self.loader = WalkLoader(
            imagenet_validation_path,
            self.data_root,
            self.seed,
            self.frequency,
            self.base_amount,
            self.accuracy,
            self.subset_size,
        )

    def _prepare_dataloader(self) -> DataLoader:
        data = self.loader.generate_dataset()
        self.targets = [s[1] for s in data]

        return DataLoader(
            sh_data_torch.IndexedTorchDataset(
                sh_data_torch.ImagesOnlyTorchDataset(data)
            ),
            max_batch_size=None,
        )

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = (all_predicted_labels == np.array(self.targets)).mean()

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )


