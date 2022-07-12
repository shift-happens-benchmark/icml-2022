"""
This task aims to evaluate models' out-of-distribution (OOD) detection on 200 raccoon images.
Raccoons are not presented in ImageNet classes, so the task uses models'
confidences (maximal predicted class probability) for the ImageNet validation set and
raccoons images (ImageNet samples treated as class 1 and raccoons as class 0) to measure
AUROC and FPR at TPR equal 0.95.

The original dataset was collected by Dat Tran for the object detection task
and can be found at https://github.com/datitran/raccoon_dataset.
"""
import dataclasses
import os

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
import torch

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data import imagenet as sh_imagenet
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.mixins import OODScoreTaskMixin
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.utils import auroc_ood
from shifthappens.tasks.utils import fpr_at_tpr
from shifthappens.tasks.ssb.imagenet_ssb import get_imagenet_ssb_dataset

@sh_benchmark.register_task(
    name="SSB", relative_data_folder="ssb", standalone=True
)
@dataclasses.dataclass
class SSB(Task, OODScoreTaskMixin):
    resource = (
        "imagenet_21k_p",
        "raccoons.tar.gz",
        "https://nc.mlcloud.uni-tuebingen.de/index.php/s/JrSQeRgXfw28crC/download/raccoons.tar.gz",
        None,
    )

    max_batch_size: int = 256

    def setup(self):
        imagenet_21k_root, ssb_split_path, url, md5 = self.resource
        dataset_folder = os.path.join(self.data_root, folder_name)
        if not os.path.exists(dataset_folder):
            sh_utils.download_and_extract_archive(url, dataset_folder, md5, file_name)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        test_transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        dataset_out = get_imagenet_ssb_dataset(imagenet21k_root=imagenet_21k_root, osr_split_path=ssb_split_path,
                                               test_transform=test_transform)

        self.images_only_dataset_out = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(dataset_out)
        )

    def _prepare_dataloader(self):
        dataloader_out = sh_data.DataLoader(
            self.images_only_dataset_out, max_batch_size=self.max_batch_size
        )
        return dataloader_out

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader_out = self._prepare_dataloader()

        ood_scores_out_list = []
        for predictions_out in model.predict(
            dataloader_out, PredictionTargets(ood_scores=True)
        ):
            assert (
                predictions_out.ood_scores is not None
            ), "OOD scores for SSB task is None"
            ood_scores_out_list.append(predictions_out.ood_scores)
        ood_scores_out = np.hstack(ood_scores_out_list)
        accuracy = np.equal(
            model.imagenet_validation_result.class_labels,
            np.array(sh_imagenet.load_imagenet_targets()),
        ).mean()  # remove for pure OOD detection
        auroc = auroc_ood(
            np.array(model.imagenet_validation_result.ood_scores), ood_scores_out
        )
        fpr_at_95 = fpr_at_tpr(
            np.array(model.imagenet_validation_result.ood_scores), ood_scores_out, 0.95
        )
        return TaskResult(
            accuracy=accuracy,
            auroc=auroc,
            fpr_at_95=fpr_at_95,
            summary_metrics={
                Metric.OODDetection: ("auroc", "fpr_at_95"),
                Metric.Robustness: "accuracy",  # remove for pure OOD detection
            },
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(
        ResNet18(device="cpu", max_batch_size=128),
        "data",
    )
