"""ImageNet-D task."""

import dataclasses
import os
from typing import Optional
from typing import Tuple

import numpy as np
import torch
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
from shifthappens.tasks.imagenet_d import map_classes
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@dataclasses.dataclass
class _ImageNetDBase(Task):
    """Base class for all ImageNet-D tasks.
    There is a subclass for each subset of the dataset.
    """

    resource: Tuple[str, ...] = abstract_variable()

    max_batch_size: Optional[int] = None

    def setup(self):
        """It build the mapping between Imangenet classes and VISDA dataset."""
        folder_name, archive_name, url, md5 = self.resource

        dataset_folder = os.path.join(self.data_root, folder_name)
        if not os.path.exists(dataset_folder):
            # download data
            sh_utils.download_and_extract_archive(
                url, self.data_root, md5, archive_name
            )

        symlinks_folder = os.path.join(self.data_root, "visda_symlinks", folder_name)
        _map = map_classes.build_map_dict(self.data_root)
        self.map_imagenet2visda = (
            map_classes.create_symlinks_and_get_imagenet_visda_mapping(
                dataset_folder, symlinks_folder, _map
            )
        )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = tv_datasets.ImageFolder(
            root=symlinks_folder, transform=test_transform
        )
        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        """Builds the DatasetLoader object."""
        return sh_data.DataLoader(
            self.images_only_dataset, max_batch_size=self.max_batch_size
        )

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluates the model on the ImageNet-D dataset."""
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)
        all_predicted_labels = self.map_imagenet2visda[all_predicted_labels]

        accuracy = torch.count_nonzero(
            all_predicted_labels.cpu() == torch.Tensor(self.ch_dataset.targets)
        ) / len(self.ch_dataset.targets)

        return TaskResult(
            accuracy=accuracy.item(), summary_metrics={Metric.Robustness: "accuracy"}
        )


@sh_benchmark.register_task(
    name="ImageNet-D (Clipart)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDClipart(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "clipart",
            "clipart.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            "cd0d8f2d77a4e181449b78ed62bccf1e",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-D (Infograph)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDInfograph(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "infograph",
            "infograph.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
            "720380b86f9e6ab4805bb38b6bd135f8",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-D (Painting)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDPainting(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "painting",
            "painting.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
            "1ae32cdb4f98fe7ab5eb0a351768abfd",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-D (Quickdraw)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDQuickdraw(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "quickdraw",
            "quickdraw.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
            "bdc1b6f09f277da1a263389efe0c7a66",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-D (Real)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDReal(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "real",
            "real.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
            "dcc47055e8935767784b7162e7c7cca6",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-D (Sketch)", relative_data_folder="imagenet_d", standalone=True
)
@dataclasses.dataclass
class ImageNetDSketch(_ImageNetDBase):
    """ImageNet-D subset"""

    resource: Tuple[str, ...] = variable(
        (
            "sketch",
            "sketch.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
            "658d8009644040ff7ce30bb2e820850f",
        )
    )
