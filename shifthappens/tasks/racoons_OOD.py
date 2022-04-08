import sys, os
import torchvision
sys.path.insert(0, os.path.join(os.environ["JULIAN"], 'icml-2022'))
import shifthappens
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
import torch
import numpy as np
from sklearn import metrics
import dataclasses
from abc import ABC
from abc import abstractmethod
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar
import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import abstract_variable
from shifthappens.tasks.base import parameter
from shifthappens.tasks.base import Task
from shifthappens.tasks.base import variable
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult
from shifthappens.models.torchvision import resnet18
class RaccOOD(Task):
    def __init__(self, device=None):
        super().__init__(data_root=None)
        self.imagenet_val_dataset_path = '/scratch/datasets/imagenet' #how do we provide datasets?
        self.racoons_dataset_path = '/scratch/jbitterwolf98/datasets/raccoons' #how should the data be submitted?
        self.batch_size = 64
        self.device = device
        
    def setup(self):
        in_dataset_folder = self.imagenet_val_dataset_path
        out_dataset_folder = self.racoons_dataset_path
        # if not os.path.exists(dataset_folder):
        #     # download data
        #     for file_name, url, md5 in self.resources:
        #         sh_utils.download_and_extract_archive(
        #             url, self.data_root, md5, file_name
        #         )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )


        self.dataset_in = tv_datasets.ImageFolder(
            root=in_dataset_folder, transform=test_transform
        )
        self.images_only_dataset_in = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.dataset_in)
        )
        
        self.dataset_out = tv_datasets.ImageFolder(
            root=out_dataset_folder, transform=test_transform
        )
        self.images_only_dataset_out = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.dataset_out)
        )

    @classmethod
    def auroc(cls, values_in, values_out):  #should be provided by an API module
        if len(values_in)*len(values_out) == 0:
            return np.NAN
        y_true = len(values_in)*[1] + len(values_out)*[0]
        y_score = np.concatenate([np.nan_to_num(-values_in, nan=0.0), np.nan_to_num(-values_out, nan=0.0)])
        return metrics.roc_auc_score(y_true, y_score)
    
    @classmethod
    def fpr_at_tpr(cls, values_in, values_out, tpr):
        if len(values_in)*len(values_out) == 0:
            return np.NAN
        t = np.quantile(values_in, (1-tpr))
        fpr = (values_out >= t).mean()
        return fpr
     
    @classmethod
    def fpr_at_95(cls, values_in, values_out):
        return cls.fpr_at_tpr(values_in, values_out, 0.95)
    
    def _prepare_dataloader_in(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset_in, max_batch_size=None)
    
    def _prepare_dataloader_out(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset_out, max_batch_size=None)
    
    def _prepare_dataloader(self):
        return self._prepare_dataloader_in()
    
    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader_in = self._prepare_dataloader_in()
        dataloader_out = self._prepare_dataloader_out()
        
        predicted_classes_in_list = [] #remove for pure OOD detection
        ood_scores_in_list = []
        for predictions_in in model.predict(
            dataloader_in, PredictionTargets(ood_scores=True, class_labels=True)
        ):
            predicted_classes_in_list.append(predictions_in.class_labels) #remove for pure OOD detection
            ood_scores_in_list.append(predictions_in.ood_scores)
        predicted_classes_in = np.concatenate(predicted_classes_in_list, 0) #remove for pure OOD detection
        ood_scores_in = np.concatenate(ood_scores_in_list, 0)
        
        ood_scores_out_list = []
        for predictions_out in model.predict(
            dataloader_out, PredictionTargets(ood_scores=True)
        ):
            ood_scores_out_list.append(predictions_out.ood_scores)
        ood_scores_out = np.concatenate(ood_scores_out_list, 0)
        
        accuracy = np.equal(predicted_classes_in, np.array(self.dataset_in.targets)).mean() #remove for pure OOD detection
        auroc = self.auroc(ood_scores_in, ood_scores_out)
        fpr_at_95 = self.fpr_at_95(ood_scores_in, ood_scores_out)

        return TaskResult(
            accuracy=accuracy,
            auroc=auroc,
            fpr_at_95=fpr_at_95,
            summary_metrics={Metric.OODDetection: ("auroc", "fpr_at_95"),
                             Metric.Robustness: "accuracy" #remove for pure OOD detection
                            }
        )
r = RaccOOD()
model = resnet18()
r.setup()
raccoon_result = r.evaluate(model)
print(raccoon_result._metrics)