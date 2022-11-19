"""Semantic shift benchmark."""

import dataclasses
import os
import pickle

import numpy as np
import requests
import torchvision.transforms as tv_transforms

import shifthappens.config
import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
from shifthappens import benchmark as sh_benchmark
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import abstract_variable
from shifthappens.tasks.base import Task
from shifthappens.tasks.base import variable
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.mixins import OODScoreTaskMixin
from shifthappens.tasks.ssb.imagenet_ssb import _get_imagenet_ssb_subset
from shifthappens.tasks.ssb.imagenet_ssb import assert_data_downloaded
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.utils import auroc_ood
from shifthappens.tasks.utils import fpr_at_tpr


@dataclasses.dataclass
class _SSB(Task, OODScoreTaskMixin):
    """
    Prepares the ImageNet evaluation from the Semantic Shift Benchmark for open-set recognition (OSR)

    Downloads SSB OSR splits to Task.data_root
    Assumes ImageNet-21KP validation splits are downloaded to shifthappens.config.imagenet21k_preprocessed_validation_path
    To download the ImageNet21k-P data:
            Follow instructions at https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md
            Ensure data is from the Winter21 ImageNet release!
    """

    OSR_URL = "https://github.com/sgvaze/osr_closed_set_all_you_need/raw/main/data/open_set_splits/imagenet_osr_splits_winter21.pkl"

    subset_type: str = abstract_variable()

    max_batch_size: int = 256

    def setup(self):
        """Asserts data is downloaded and sets up open-set dataset"""
        osr_split_path = os.path.join(
            self.data_root, "imagenet_osr_splits_winter21.pkl"
        )
        if not os.path.exists(osr_split_path):
            os.makedirs(self.data_root, exist_ok=True)
            osr_split = requests.get(self.OSR_URL)
            open(osr_split_path, "wb").write(osr_split.content)
        else:
            with open(osr_split_path, "rb") as f:
                osr_split = pickle.load(f)
        # Ensure data is downloaded
        assert_data_downloaded(
            osr_split, shifthappens.config.imagenet21k_preprocessed_validation_path
        )
        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        dataset_out = _get_imagenet_ssb_subset(
            imagenet21k_root=shifthappens.config.imagenet21k_preprocessed_validation_path,
            osr_split=osr_split,
            test_transform=test_transform,
            subset_type=self.subset_type,
        )

        self.dataset_out = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(dataset_out)
        )

    def _prepare_dataloader(self):
        dataloader_out = sh_data.DataLoader(
            self.dataset_out, max_batch_size=self.max_batch_size
        )
        return dataloader_out

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()
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


@sh_benchmark.register_task(
    name="SSB_easy", relative_data_folder="ssb", standalone=True
)
@dataclasses.dataclass
class SSBEasy(_SSB):
    """SSB Easy subset"""

    subset_type: str = variable("easy")


@sh_benchmark.register_task(
    name="SSB_hard", relative_data_folder="ssb", standalone=True
)
@dataclasses.dataclass
class SSBHard(_SSB):
    """SSB Hard subset"""

    subset_type: str = variable("hard")
