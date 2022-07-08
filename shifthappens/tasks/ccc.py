"""
Implementation of CCC: Continuously Changing Corruptions
Note: This Task only implements the data reading portion of the dataset.
In addition to this file, we submitted a file used to generate the data itself.
"""

import dataclasses
import os

import numpy as np

import shifthappens.data.base as sh_data
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.base import parameter

from shifthappens.tasks.ccc_utils import WalkLoader


@sh_benchmark.register_task(
    name="CCC", relative_data_folder="ccc", standalone=True
)
@dataclasses.dataclass
class CCC(Task):
    seed: int = parameter(
        description="random seed used in the dataset building process",
    )
    frequency: int = parameter(
        description="represents how many images are sampled from each subset",
    )
    base_amount: int = parameter(
        description="represents how large the base dataset is",
    )
    accuracy: int = parameter(
        description="represents the baseline accuracy of walk",
    )
    subset_size: int = parameter(
        description="represents the sample size of images sampled from ImageNet validation",
    )

    def setup(self):
        self.loader = WalkLoader(os.path.join(self.data_root, "ccc"), './ccc_accuracy_matrix.pickle',
                                 self.seed, self.frequency, self.base_amount, self.accuracy, self.subset_size)

    def _prepare_dataloader(self) -> DataLoader:
        self.setup()
        data = self.loader.get()
        return sh_data.DataLoader(data, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
                dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = all_predicted_labels == np.array(self.ch_dataset.targets)

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet50
    sh_benchmark.evaluate_model(ResNet50(), "data")
