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
from shifthappens.tasks.base import variable
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult
from examples.imagenet_d_helper import map_classes

@dataclasses.dataclass
class ImageNetSingleCorruptionTypeBase(Task):
    resource: Tuple[str, ...] = abstract_variable()
    
    max_batch_size: Optional[int] = None

    def setup(self):
        folder_name, archive_name, url, md5 = self.resource

        self.data_root="examples/test_data/imagenet_d" # TODO Remove

        dataset_folder = os.path.join(self.data_root, folder_name)
        if not os.path.exists(dataset_folder):
            # download data
            sh_utils.download_and_extract_archive(
                url, self.data_root, md5, archive_name
            )

        symlinks_folder = os.path.join(self.data_root, "visda_symlinks", folder_name)
        _map = map_classes.build_map_dict()
        self.map_imagenet2visda = \
            map_classes.create_symlinks_and_get_imagenet_visda_mapping(
                dataset_folder, symlinks_folder, _map)

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

        all_predicted_labels = self.map_imagenet2visda[all_predicted_labels]
        
        accuracy = (all_predicted_labels == np.array(self.ch_dataset.targets))

        return TaskResult(
            accuracy=accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


# noise corruptions
@sh_benchmark.register_task(
    name="ImageNet-D (Clipart)",
    relative_data_folder="imagenet_d",
    standalone=True,
)
@dataclasses.dataclass
class ImageNetDClipart(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "clipart",
            "clipart.zip",
            "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
            "cd0d8f2d77a4e181449b78ed62bccf1e",
        )
    )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cpu", max_batch_size=128), "test_data")
