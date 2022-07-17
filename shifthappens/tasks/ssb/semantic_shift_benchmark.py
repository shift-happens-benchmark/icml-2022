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
import pickle

import numpy as np
import torchvision.transforms as tv_transforms
import torch

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.mixins import OODScoreTaskMixin
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.utils import auroc_ood
from shifthappens.tasks.utils import fpr_at_tpr
from shifthappens.tasks.ssb.imagenet_ssb import get_imagenet_ssb_datasets
from shifthappens.tasks.ssb import osr_split_path

from typing import Tuple

@sh_benchmark.register_task(
    name="SSB", relative_data_folder="ssb", standalone=True
)
@dataclasses.dataclass
class SSB(Task, OODScoreTaskMixin):

    resource = (
        "imagenet21k_resized_new",
        osr_split_path,
        None,
        None,
    )
    dataset_out_easy: sh_data_torch.IndexedTorchDataset = None
    dataset_out_hard: sh_data_torch.IndexedTorchDataset = None

    max_batch_size: int = 256

    def setup(self):

        imagenet_21k_dir_name, ssb_split_path, url, md5 = self.resource
        imagenet_21k_root = os.path.join(self.data_root, imagenet_21k_dir_name)

        # Ensure data is downloaded
        assert (
            self.assert_data_downloaded(imagenet_21k_root)
        ), f'ImageNet-21K data not downloaded not found in {imagenet_21k_root} \n Please download and process according to:' \
           ' https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh'

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        # test_transform = tv_transforms.Compose([
        #     tv_transforms.Resize(256),
        #     tv_transforms.CenterCrop(224),
        #     tv_transforms.ToTensor(),
        #     tv_transforms.Normalize(
        #         mean=torch.tensor(mean),
        #         std=torch.tensor(std))
        # ])

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        dataset_out_easy, dataset_out_hard = get_imagenet_ssb_datasets(imagenet21k_root=imagenet_21k_root,
                                                                        osr_split_path=ssb_split_path,
                                                                        test_transform=test_transform)

        self.dataset_out_easy = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(dataset_out_easy)
        )

        self.dataset_out_hard = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(dataset_out_hard)
        )

    def _prepare_dataloader(self):
        dataloader_out_easy = sh_data.DataLoader(
            self.dataset_out_easy, max_batch_size=self.max_batch_size
        )

        dataloader_out_hard = sh_data.DataLoader(
            self.dataset_out_hard, max_batch_size=self.max_batch_size
        )

        return dataloader_out_easy, dataloader_out_hard

    @staticmethod
    def _evaluate_single_split(model: sh_models.Model, dataloader: sh_data.DataLoader) -> TaskResult:

        ood_scores_out_list = []
        for predictions_out in model.predict(
                dataloader, PredictionTargets(ood_scores=True)
        ):
            assert (
                    predictions_out.ood_scores is not None
            ), "OOD scores for SSB task is None"
            ood_scores_out_list.append(predictions_out.ood_scores)
        ood_scores_out = np.hstack(ood_scores_out_list)

        auroc = auroc_ood(
            np.array(model.imagenet_validation_result.ood_scores), ood_scores_out
        )
        fpr_at_95 = fpr_at_tpr(
            np.array(model.imagenet_validation_result.ood_scores), ood_scores_out, 0.95
        )
        return TaskResult(
            auroc=auroc,
            fpr_at_95=fpr_at_95,
            summary_metrics={
                Metric.OODDetection: ("auroc", "fpr_at_95"),
            },
        )

    def _evaluate(self, model: sh_models.Model) -> Tuple[TaskResult, TaskResult]:

        dataloader_out_easy, dataloader_out_hard = self._prepare_dataloader()
        result_easy = self._evaluate_single_split(model, dataloader_out_easy)
        result_hard = self._evaluate_single_split(model, dataloader_out_hard)

        return result_easy, result_hard

    @staticmethod
    def assert_data_downloaded(imagenet21k_root):

        try:

            # Load splits
            with open(osr_split_path, 'rb') as handle:
                precomputed_info = pickle.load(handle)

            easy_wnids = precomputed_info['easy_i21k_classes']
            hard_wnids = precomputed_info['hard_i21k_classes']
            all_wnids = easy_wnids.tolist() + hard_wnids

            dataset_folder = os.path.join(imagenet21k_root, 'val')
            paths = os.listdir(dataset_folder)

            return all([x in paths for x in all_wnids])

        except:

            return False


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(
        ResNet18(device="cpu", max_batch_size=128),
        "data",
    )
