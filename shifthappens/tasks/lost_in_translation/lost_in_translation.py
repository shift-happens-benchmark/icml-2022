"""
TODO
"""
import dataclasses
import os

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data import imagenet as sh_imagenet
from shifthappens.models import base as sh_models
from shifthappens.models import torchvision as sh_models_t
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.mixins import OODScoreTaskMixin
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.utils import auroc_ood
from shifthappens.tasks.utils import fpr_at_tpr
from shifthappens.tasks.base import parameter

import shifthappens.tasks.lost_in_translation.affine_transformations.affine as  a
import shifthappens.tasks.lost_in_translation.affine_transformations.affine_linspace as  lin
import shifthappens.tasks.lost_in_translation.affine_transformations.affine_linspace_adaptive as lin_a
import shifthappens.tasks.lost_in_translation.affine_transformations.statistics as stat

from torchvision import transforms
import shifthappens.tasks.lost_in_translation.imagenet_s.imagenet_s as i_s
import shifthappens.config
import random
import math


@sh_benchmark.register_task(
    name="LostInTranslation", relative_data_folder="lost_in_translation", standalone=True
)
@dataclasses.dataclass
class LostInTranslationBase(Task):
    """
    TODO
    """

    default_batch_size: int = 700

    resolution: int = parameter(
        default=224,
        description="resolution expected from the model",
    )

    rotation_cutoff: int = parameter(
        default=0,
        description="restrict the dataset to samples with at least these degrees of rotation",
    )

    translation_cutoff: int = parameter(
        default=0,
        description="restrict the dataset to samples with at least these x pixels translatin freedom",
    )


    resource_s = (
        "imagenet_s",
        #"raccoons.tar.gz",
        #"https://nc.mlcloud.uni-tuebingen.de/index.php/s/JrSQeRgXfw28crC/download/raccoons.tar.gz",
        None,
        None,
        None,
    )

    def setup(self):
        """Load and prepare the data."""

        folder_name_s, file_name, url, md5 = self.resource_s
        imagent_s_folder = os.path.join(self.data_root, folder_name_s)
        # if not os.path.exists(dataset_folder):
        #     sh_utils.download_and_extract_archive(url, dataset_folder, md5, file_name)

        a.config = a.config_imagenet
        a.config['target_size'] = self.resolution
        a.config['crop_size'] =  self.resolution


        tt = transforms.ToTensor()

        imagenet_root_path = shifthappens.config.imagenet_validation_path

        params = i_s.get_param('300')
        num_classes = params['num_classes']

        name_list = os.path.join(f'{imagent_s_folder}/names', params['names'])

        subdir = 'validation-segmentation'
        gt_dir = os.path.join(imagent_s_folder, params['dir'], subdir)
        self.dataset = i_s.ImageNetSEvalDataset(imagenet_root_path, gt_dir, name_list, transform=tt,
            use_new_labels=True, simple_items=True, prefilter_items=True,
            transform_mask_to_img_classes=True)

        label_map = {}
        for i in range(len(self.dataset)):
            I, gt_uint, in_label = self.dataset[i]
            label_map[i] = in_label

        stat.label_map = label_map

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        data = []
        idx_list = list(range(len(self.dataset)))
        random.shuffle(idx_list)
        for i in idx_list:
            try:
                I, gt_uint, in_label = self.dataset[i]
                loaded = a.load_and_pad_imagenet(I, gt_uint, in_label)
                element = a.to_torch(*loaded)
                data.append((element, in_label,i))
            except AssertionError as e:
                raise e
            except:
                pass

        if isinstance(model, sh_models_t.__TorchvisionModel):
            batch_size_model = model.max_batch_size
            eval_device = model.device
            m = model.model
            res = self._eval_model(m, "anon", batch_size_model, data, eval_device)
            return TaskResult(
                accuracy_base=res["base_case"],
                accuracy_worst=res["worst_case"],
                summary_metrics={
                    Metric.Robustness: "accuracy_base",
                    Metric.Robustness: "accuracy_worst",
                },
            )
        else:
            raise Exception("TODO!")

@sh_benchmark.register_task(
    name="LostInTranslationTranslation", relative_data_folder="lost_in_translation", standalone=True
)
@dataclasses.dataclass
class LostInTranslationTranslation(LostInTranslationBase):
    def _eval_model(model, model_name, batch_size_model, data, eval_device):
    
        args = (model, model_name, data, eval_device, batch_size_model)

        #print_base_worst(model, results_rotation, "rotation", "rotation")

        #results_rotation_filtered = [d for d in results_rotation if stat.gt_30(d)]

        #print_base_worst(model, results_rotation_filtered, "rotation deg>=30", "rotation")

        resolutions = [25,25,25]
        num_points=3

        adapt_grid_to_min=True

        #TODO: Make this batch_size indipendent/configurable

        # perc = 0.5
        # resolutions = [1.0,0.5,0.2]
        rec_depth = 2
        num_points = 2
        sizes=[4.6,2.5]
        sqrt_bl = math.sqrt(batch_size_model)
        resolutions = [
            2.0,
            (sizes[0] + 1) / (sqrt_bl - 1),
            (sizes[1] + 1) / (sqrt_bl - 1)]
        
        leng_center_2=2.3
        step_size_center_2=  (leng_center_2 + 1) / (math.sqrt(batch_size_model) - 1)

        results_translation = lin_a.translation_linspace_adaptive(*args, batch_size_model,
            resolutions, num_points, size_recursive=sizes, num_recursion=rec_depth, idx_fun=lambda x:x, target_zoom=0.8,
            save_dir = None, step_size_center=step_size_center_2,
            leng_center=leng_center_2, adapt_grid_to_min=True, early_stopping=True,
            find_min_correct=False, adapt_resolution=True)

        stat_translation = {
            "base_case": stat.adaptive_base_case(results_translation, "trans"),
            "worst_case": stat.adaptive_worst_case(results_translation, "trans"),
        }

        return stat_translation

@sh_benchmark.register_task(
    name="LostInTranslationRotation", relative_data_folder="lost_in_translation", standalone=True
)
@dataclasses.dataclass
class LostInTranslationRotation(LostInTranslationBase):
    def _eval_model(model, model_name, batch_size_model, data, eval_device):
    
        args = (model, model_name, data, eval_device, batch_size_model)


        results_rotation = lin.rotation_linspace(*args, batch_size_rotation=batch_size_model, resolution=200, do_resize=False, save_dir=None)


        stat_rotation = {
            "base_case": stat.adaptive_base_case(results_rotation, "rotation"),
            "worst_case": stat.adaptive_worst_case(results_rotation, "rotation"),
        }

        return stat_rotation