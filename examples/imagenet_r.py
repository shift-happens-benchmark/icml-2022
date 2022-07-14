"""A Task for evaluating the classification accuracy on ImageNet-R."""

import dataclasses
import os

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="ImageNet-R", relative_data_folder="imagenet_r", standalone=True
)
@dataclasses.dataclass
class ImageNetR(Task):
    """Measures the classification accuracy on ImageNet-R [1], a dataset
    containing different renditions of 200 classes of ImageNet (30000 samples in total).

    [1] The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization.
        Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai,
        Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt and Justin Gilmer. 2021.
    """

    resources = [
        (
            "imagenet-r.tar",
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            "A61312130A589D0CA1A8FCA1F2BD3337",
        )
    ]

    def setup(self):
        """Load and prepare the data."""

        dataset_folder = os.path.join(self.data_root, "imagenet-r")
        if not os.path.exists(dataset_folder):
            # download data
            for file_name, url, md5 in self.resources:
                sh_utils.download_and_extract_archive(
                    url, self.data_root, md5, file_name
                )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = tv_datasets.ImageFolder(
            root=dataset_folder, transform=test_transform
        )
        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

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
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cpu", max_batch_size=128), "data")
