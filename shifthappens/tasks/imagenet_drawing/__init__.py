"""Shift Happens task: ImageNet-Drawing"""

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
    name="ImageNet-Drawing", relative_data_folder="imagenet_drawing", standalone=True
)
@dataclasses.dataclass
class ImageNetDrawing(Task):
    """ImageNet-Drawing Dataset.
    This task evaluates a model on ImageNet-Drawing. This 
    dataset was formed by converting the images in the
    ImageNet validation set into colored pencil drawings
    using simple image processing. See the readme file for
    more information about how the dataset was constructed.
    The goal of this evaluation task is to measure the
    model's robustness to distribution shifts.
    """
    
    resources = [
        (
            "imagenet-drawing.tar.gz",
            "https://zenodo.org/record/6801109/files/imagenet-drawing.tar.gz?download=1",
            "3fb1206b6e3190d0159e5dc01c0f97ab",
        )
    ]

    def setup(self):
        """Setup ImageNet-Drawing"""
        dataset_folder = os.path.join(self.data_root, "imagenet-drawing")
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
        """Builds the DatasetLoader object."""
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluates the model on the ImageNet-Drawing dataset."""
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
