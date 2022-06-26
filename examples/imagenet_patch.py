import dataclasses
import os
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
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@dataclasses.dataclass
class ImageNetPatchTarget(Task):
    resource: Tuple[str, ...] = ('imagenet_patch', 'ImageNet-Patch.tar.gz',
                                 'https://zenodo.org/record/6568778/files/ImageNet-Patch.gz?download=1',
                                 'a9d9cc4d77d2a192b3386118b70422c2')

    target: int = abstract_variable()
    max_batch_size: Optional[int] = None

    def setup(self):
        folder_name, archive_name, url, md5 = self.resource

        dataset_folder = os.path.join(self.data_root, "Imagenet-Patch")
        if not os.path.exists(dataset_folder):
            # download data
            sh_utils.download_and_extract_archive(
                url, self.data_root, md5, archive_name
            )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        dataset_folder = os.path.join(dataset_folder, str(self.target))
        self.ch_dataset = tv_datasets.ImageFolder(
            root=dataset_folder, transform=test_transform
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
        )
        return TaskResult(
            accuracy=accuracy,
            mce=1.0 - accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


# patch corruptions
@sh_benchmark.register_task(
    name="ImageNet-Patch (banana)",
    relative_data_folder="imagenet_patch",
    standalone=True,
)
@dataclasses.dataclass
class ImageNetPatchBanana(ImageNetPatchTarget):
    target: int = 954


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cpu", max_batch_size=128), "test_data")
