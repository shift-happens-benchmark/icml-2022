"""SI-Score dataset.
https://github.com/google-research/si-score

The dataset measures robustness of image classification models with
ImageNet class output to changes in object size, location and
rotation angle.
"""

import dataclasses
import os
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

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
from shifthappens.tasks.base import abstract_variable
from shifthappens.tasks.base import Task
from shifthappens.tasks.base import variable
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.siscore import siscore_labels
from shifthappens.tasks.task_result import TaskResult

_BASE_URL = "https://s3.us-east-1.amazonaws.com/si-score-dataset"

_VARIANT_EXPANDED_DIR_NAMES = {
    "size": "area",
    "rotation": "rotation",
    "location": "location20_area02_min0pc",
}


class SISCOREImageFolder(tv_datasets.ImageFolder):
    """Version of ImageFolder that converts SI-Score text labels to ImageNet int labels."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
        )
        # Create dict mapping dataset labels to ImageNet classes.
        self.idx_to_imagenet_idx = {}
        for class_name, label_idx in self.class_to_idx.items():
            imagenet_class = siscore_labels.IMAGENET_LABELS[class_name]
            self.idx_to_imagenet_idx[label_idx] = imagenet_class

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        # Map labels to ImageNet labels
        target = self.idx_to_imagenet_idx[target]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


@dataclasses.dataclass
class SISCOREVariantBase(Task):
    """Classification task on synthetic data to measure a model's robustness to changes in
    object location, size or rotation angle.
    """

    resource: Tuple[str, ...] = abstract_variable()

    max_batch_size: Optional[int] = None

    def setup(self):
        """Setup dataset/task."""
        variant = self.resource
        url = f"{_BASE_URL}/{variant}.zip"

        dataset_folder = os.path.join(
            self.data_root, _VARIANT_EXPANDED_DIR_NAMES[variant]
        )
        if not os.path.exists(dataset_folder):
            # Download data.
            sh_utils.download_and_extract_archive(
                url, self.data_root, md5=None, filename=f"{variant}.zip"
            )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = SISCOREImageFolder(
            root=dataset_folder,
            transform=test_transform,
        )
        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        return sh_data.DataLoader(
            self.images_only_dataset, max_batch_size=self.max_batch_size
        )

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = (
            all_predicted_labels
            == np.array(self.ch_dataset.targets)[: len(all_predicted_labels)]
        ).mean()

        print(f"Accuracy: {np.mean(accuracy)}")
        return TaskResult(
            accuracy=accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


# Varying object size.
@sh_benchmark.register_task(
    name="SISCORE (Size)",
    relative_data_folder="siscore",
    standalone=True,
)
@dataclasses.dataclass
class SISCORESize(SISCOREVariantBase):
    """Various object size subset."""

    resource: Tuple[str, ...] = variable(("size"))


# Varying object rotation angle.
@sh_benchmark.register_task(
    name="SISCORE (Rotation)",
    relative_data_folder="siscore",
    standalone=True,
)
@dataclasses.dataclass
class SISCORERotation(SISCOREVariantBase):
    """Various rotations subset."""

    resource: Tuple[str, ...] = variable(("rotation"))


# Varying object location.
@sh_benchmark.register_task(
    name="SISCORE (Location)",
    relative_data_folder="siscore",
    standalone=True,
)
@dataclasses.dataclass
class SISCORELocation(SISCOREVariantBase):
    """Various locations subset."""

    resource: Tuple[str, ...] = variable(("location"))
