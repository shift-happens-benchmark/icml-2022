"""A Shift Happens task: Model-aware/agnostic Adversarial Patch Attack on ImageNet"""

import dataclasses
import os

import numpy as np
import torch
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms
import torch.nn as nn

import sys
sys.path.append('./')

import shifthappens.models.base as sh_models
import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.utils as sh_utils

from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.base import parameter
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="ImageNet_Adv_Patch", relative_data_folder="imagenet_val/", standalone=True
)


@dataclasses.dataclass
class ImageNet_Adv_Patch(Task):

    attack_iter: int = parameter(
        default=1000,
        options=(7, 10, 20, 40, 100, 200, 500, 1000, 10000),
        description="the number of attack iterations",
    )

    attack_stepsize: float = parameter(
        default=0.001,
        options=(0.001, 0.01),
        description="the step size of adversarial patch attack",
    )

    batch_size: int = parameter(
        default=256,
        options=(256, 512, 1024),
        description="the number of attack iterations",
    )

    patch_size: int = parameter(
        default=16,
        options=(16, 32, 64),
        description="the number of attack iterations",
    )


    def setup(self):
        dataset_folder = os.path.join(self.data_root, "./")
        if not os.path.exists(dataset_folder):
            print('The specified data path does not exist. Please Specify the data path correctly !!!')
            exit()

        test_transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

        self.ch_dataset = tv_datasets.ImageFolder(
            root=dataset_folder, transform=test_transform
        )


    def _prepare_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.ch_dataset, batch_size=self.batch_size, shuffle=True)

    def _clip(self, batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        batch[:, 0, :, :] = batch[:, 0, :, :] * std[0] + mean[0]
        batch[:, 1, :, :] = batch[:, 1, :, :] * std[1] + mean[1]
        batch[:, 2, :, :] = batch[:, 2, :, :] * std[2] + mean[2]

        batch = torch.clamp(batch, min=0.0, max=1.0)

        batch[:, 0, :, :] = (batch[:, 0, :, :] - mean[0]) / std[0]
        batch[:, 1, :, :] = (batch[:, 1, :, :] - mean[1]) / std[1]
        batch[:, 2, :, :] = (batch[:, 2, :, :] - mean[2]) / std[2]
        return batch

    def _get_mask(self, batch_images):
        mask = torch.zeros_like(batch_images)
        input_size = mask.shape[-1]
        row, col = np.random.randint(input_size-self.patch_size, size=2)
        mask[:, :, row:self.patch_size+row, col:self.patch_size+col] = 1
        return mask


    def _get_adv_patch_images(self, model, batch, device):
        adv_batch_preprocessed = batch[0].to(device)  # model._pre_process(batch, device)
        adv_batch_preprocessed.requires_grad = True 

        targets = batch[1].to(device)
        patch_mask = self._get_mask(batch[0]).to(device)

        criterion = nn.CrossEntropyLoss(reduction="sum")
        for i in range(self.attack_iter):
            outputs = model.model(adv_batch_preprocessed)
            loss    = criterion(outputs, targets)
            loss.backward()

            grad = adv_batch_preprocessed.grad
            adv_batch_preprocessed.data = self._clip(adv_batch_preprocessed + patch_mask * self.attack_stepsize * torch.sign(grad)).detach()
            adv_batch_preprocessed.grad.zero_()
            model.model.zero_grad()
        return adv_batch_preprocessed


    def _eval_preprocessed_batch(self, model, preprocessed_batch, device):
        logits, features = model.hooked_model(preprocessed_batch)

        features = features.view(len(features), -1).detach().cpu()
        probabilities = torch.softmax(logits, -1).detach().cpu()

        max_confidences, predictions = probabilities.max(1)
        return sh_models.ModelResult(class_labels=predictions.numpy(),
                confidences=probabilities.numpy(),
                ood_scores=max_confidences.numpy(),
                features=features.numpy(),
            )


    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        device = model.device
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for batch in dataloader:
            processed_batch = self._get_adv_patch_images(model, batch, device)
            predictions = self._eval_preprocessed_batch(model, processed_batch, device)
            all_predicted_labels_list.append(predictions.class_labels)

        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = all_predicted_labels == np.array(self.ch_dataset.targets)

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cuda:0", max_batch_size=8), "../")
