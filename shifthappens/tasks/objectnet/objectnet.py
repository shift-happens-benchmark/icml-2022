"""
Shift Happens task: ObjectNet

What is ObjectNet?
A new kind of vision dataset borrowing the idea of controls from other areas of science.
No training set, only a test set! Put your vision system through its paces.
Collected to intentionally show objects from new viewpoints on new backgrounds.
50,000 image test set, same as ImageNet, with controls for rotation, background, and viewpoint.
313 object classes with 113 overlapping ImageNet
Large performance drop, what you can expect from vision systems in the real world!
Robust to fine-tuning and a very difficult transfer learning problem.

This task aims to measure robustness of the model on the ObjectNet subset -
intersection of ObjectNet and ImageNet classes.
"""

import dataclasses
import os

import numpy as np
import pandas as pd
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.objectnet import objectnet_utils
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="ObjectNet", relative_data_folder="objectnet", standalone=True
)
@dataclasses.dataclass
class ObjectNet(Task):
    """
    The class that wraps evaluation of models' robustness on the ObjectNet
    subset intersecting with ImageNet classes.
    """

    imagenet_to_objectnet_json_url = (
        "https://raw.githubusercontent.com"
        "/abarbu/objectnet-template-tensorflow"
        "/master/mapping_files"
        "/imagenet_id_to_objectnet_id.json"
    )

    resources = [
        (
            "objectnet-1.0.zip",
            "https://objectnet.dev/downloads/objectnet-1.0.zip",
            "522aa986a604c41ea641411c1d0a292e",
            "objectnetisatestset",
        )
    ]

    def setup(self):
        """Setup ObjectNet"""
        dataset_folder = os.path.join(self.data_root, "objectnet-1.0")
        if not os.path.exists(dataset_folder):
            # download data
            for file_name, url, md5, password in self.resources:
                objectnet_utils.download_and_extract_zip_with_pwd(
                    url, self.data_root, md5, file_name, password
                )
        mapping_path = os.path.join(
            self.data_root, "objectnet-1.0", "objectnet-1.0", "mappings"
        )
        json_path = os.path.join(mapping_path, "imagenet_id_to_objectnet_id.json")

        if not os.path.exists(json_path):
            mapping_json = pd.read_json(
                self.imagenet_to_objectnet_json_url, typ="series"
            )
            mapping_json.to_json(json_path)
        # ImageNet -> ObjectNet mapping
        self.mapping_json = pd.read_json(json_path, typ="series").to_dict()

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.Pad(
                    -2
                ),  # every image in ObjectNet has 1px red border (check for details https://objectnet.dev/download.html);
                # however, we printed image with tv_transforms.Pad(-1) and
                # border remained that's why we use -2px here
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        images_path = os.path.join(dataset_folder, "objectnet-1.0", "images")

        self.ch_dataset = objectnet_utils.ImageFolderImageNetClassesIntersection(
            mapping_path, root=images_path, transform=test_transform
        )

        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        """Builds the DataLoader object."""
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """Evaluates the model on the subset of ObjectNet dataset."""
        dataloader = self._prepare_dataloader()
        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)

        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = objectnet_utils.class_intersection_accuracy(
            np.array(self.ch_dataset.targets), all_predicted_labels, self.mapping_json
        )
        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )
