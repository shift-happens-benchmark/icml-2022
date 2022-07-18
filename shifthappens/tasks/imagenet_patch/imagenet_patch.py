"""A task for evaluating the classification accuracy on ImageNet-Patch.

The dataset is constructed with 5000 images from the validation set of ImageNet.
The task allows loading the 5000 images with the applied patches from one or
more of the target labels defined.

"""

import dataclasses
import os
import typing
from typing import Optional
from typing import Tuple

import numpy as np
import torchvision.transforms as tv_transforms

import shifthappens
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
from shifthappens.tasks.imagenet_patch.imagenet_patch_utils import (
    ImageFolderWithEmptyDirs,
)
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@dataclasses.dataclass
class ImageNetPatchTarget(Task):
    """Class that wraps patches optimized to cause targeted misclassifications
    towards a specific class (e.g., banana)."""

    resource: Tuple[str, ...] = (
        "imagenet_patch",
        "ImageNet-Patch.tar.gz",
        "https://zenodo.org/record/6568778/files/ImageNet-Patch.gz?download=1",
        "a9d9cc4d77d2a192b3386118b70422c2",
    )

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
        self.ch_dataset = ImageFolderWithEmptyDirs(
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
        all_preds = np.concatenate(all_predicted_labels_list, 0)

        accuracy = all_preds == np.array(self.ch_dataset.targets)[: len(all_preds)]
        return TaskResult(
            accuracy=accuracy,
            mce=1.0 - accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


@sh_benchmark.register_task(
    name="ImageNet-Patch (cellular telephone)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchCellularTelephone(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `cellular telephone` class."""

    target: int = 487


@sh_benchmark.register_task(
    name="ImageNet-Patch (cornet)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchCornet(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `cornet` class."""

    target: int = 513


@sh_benchmark.register_task(
    name="ImageNet-Patch (electric guitar)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchElectricGuitar(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `electric guitar` class."""

    target: int = 546


@sh_benchmark.register_task(
    name="ImageNet-Patch (hair spray)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchHairSpray(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `hair spray` class."""

    target: int = 585


@sh_benchmark.register_task(
    name="ImageNet-Patch (soap dispenser)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchSoapDispenser(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `soap dispenser` class."""

    target: int = 804


@sh_benchmark.register_task(
    name="ImageNet-Patch (sock)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchSock(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `sock` class."""

    target: int = 806


@sh_benchmark.register_task(
    name="ImageNet-Patch (typewriter keyboard)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchTypewriterKeyboard(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `typewriter keyboard` class."""

    target: int = 878


@sh_benchmark.register_task(
    name="ImageNet-Patch (plate)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchPlate(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `plate` class."""

    target: int = 923


@sh_benchmark.register_task(
    name="ImageNet-Patch (banana)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchBanana(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `banana` class."""

    target: int = 954


@sh_benchmark.register_task(
    name="ImageNet-Patch (cup)",
    relative_data_folder="imagenet_patch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetPatchCup(ImageNetPatchTarget):
    """Class loading the images with applied patch targeting the
    `cup` class."""

    target: int = 968


@sh_benchmark.register_task(
    name="ImageNet-Patch", relative_data_folder="imagenet_patch", standalone=True
)
@dataclasses.dataclass
class ImageNetPatchCorruptions(Task):
    """Classification task on the ImageNet-Patch dataset where models might use all images
    containing all the target patches.
    """

    max_batch_size: Optional[int] = None

    corruption_task_cls: Tuple[typing.Type[ImageNetPatchTarget], ...] = variable(
        (
            ImageNetPatchCellularTelephone,
            ImageNetPatchCornet,
            ImageNetPatchElectricGuitar,
            ImageNetPatchHairSpray,
            ImageNetPatchSoapDispenser,
            ImageNetPatchSock,
            ImageNetPatchTypewriterKeyboard,
            ImageNetPatchPlate,
            ImageNetPatchBanana,
            ImageNetPatchCup,
        )
    )

    flavored_corruption_tasks: typing.List[ImageNetPatchTarget] = variable([])

    def setup(self):
        for corruption_task_cls in self.corruption_task_cls:
            self.flavored_corruption_tasks += list(
                corruption_task_cls.iterate_flavours(data_root=self.data_root)
            )
        for flavored_corruption_task in self.flavored_corruption_tasks:
            flavored_corruption_task.setup()

    def _prepare_dataloader(self) -> Optional[DataLoader]:
        return None

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        results = {}
        accuracies, mces = [], []
        for flavored_corruption_task in self.flavored_corruption_tasks:
            dl = flavored_corruption_task._prepare_dataloader()
            if dl is not None:
                model.prepare(dl)
            corruption_result = flavored_corruption_task.evaluate(model)
            if corruption_result is None:
                # model is not compatible with a subtask and the result should be ignored
                continue

            corruption_name = getattr(
                flavored_corruption_task,
                shifthappens.task_data.task_metadata._TASK_METADATA_FIELD,
            )  # type: ignore
            results[f"accuracy_{corruption_name}"] = corruption_result["accuracy"]
            results[f"mCE_{corruption_name}"] = corruption_result["mce"]

            accuracies.append(corruption_result.accuracy)
            mces.append(corruption_result.mce)

        return TaskResult(
            **results,
            accuracy=np.mean(accuracies).item(),
            mce=np.mean(mces).item(),
            summary_metrics={Metric.Robustness: "accuracy"},
        )
