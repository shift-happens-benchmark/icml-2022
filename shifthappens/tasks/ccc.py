"""Example for a Shift Happens task: CCC"""

import dataclasses
import os
import random
import itertools
import pickle
import torch

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="CCC", relative_data_folder="ccc", standalone=True
)
@dataclasses.dataclass
class CCC(Task):
    def setup(self, freq, seed, accuracy, base_amount):
        self.dataset_folder = os.path.join(self.data_root, "ccc")
        self.accuracy_dict = pickle.load(os.path.join(self.data_root, "ccc", "accuracies"))

        self.freq = freq
        self.seed = seed
        self.accuracy = accuracy
        self.base_amount = base_amount

        accuracy_dict = {}
        walk_dict = {}
        self.single_noises = [
            'gaussian_noise',
            'shot_noise',
            'impulse_noise',
            'defocus_blur',
            'glass_blur',
            'motion_blur',
            'zoom_blur',
            'snow',
            'frost',
            'fog',
            'brightness',
            'contrast',
            'elastic',
            'pixelate',
            'jpeg'
        ]

        noise_list = list(itertools.product(self.single_noises, self.single_noises))

        for i in range(len(noise_list)):
            noise1, noise2 = noise_list[i]
            if noise1 == noise2:
                continue
            noise1 = noise1.lower().replace(" ", "_")
            noise2 = noise2.lower().replace(" ", "_")

            accuracy_matrix = np.load(self.accuracy_dict)["n1_" + noise1 + "_n2_"]
            walk = find_path(accuracy_matrix, self.base_accuracy)

            accuracy_dict[(noise1, noise2)] = accuracy_matrix
            walk_dict[(noise1, noise2)] = walk

        keys = list(accuracy_dict.keys())
        cur_noises = random.choice(keys)
        noise1 = cur_noises[0].lower().replace(" ", "_")
        noise2 = cur_noises[1].lower().replace(" ", "_")

        walk = walk_dict[cur_noises]
        data_path = os.path.join(self.dataset_folder, "n1_" + noise1 + "_n2_" + noise2)
        walk_datasets = path_to_dataset(walk, data_path)

        self.walk_dict = walk_dict
        self.walk_ind = 0
        self.walk = walk
        self.walk_datasets = walk_datasets
        self.noise1 = noise1
        self.first_noise1 = self.noise1
        self.noise2 = noise2

        self.lifetime_total = 0
        self.lastrun = 0

        self.transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

    def _prepare_dataloader(self) -> DataLoader:
        all_data = None

        while True:
            path = self.walk_datasets[self.walk_ind]
            cur_data = tv_datasets.ImageFolder(path, transform=self.transform)
            inds = np.random.permutation(len(cur_data))[:self.freq]
            cur_data = torch.utils.data.Subset(cur_data, inds)

            if all_data is not None:
                all_data = torch.utils.data.ConcatDataset([all_data, cur_data])
            else:
                all_data = cur_data

            self.lifetime_total += self.freq

            if self.walk_ind == len(self.walk) - 1:
                self.noise1 = self.noise2

                if self.lifetime_total > self.base_amount and self.lastrun == 0:
                    if self.noise1 != self.first_noise1:
                        self.noise2 = self.first_noise1
                        self.lastrun = 1
                    else:
                        break
                elif self.lastrun == 1:
                    break
                else:
                    while self.noise1 == self.noise2:
                        self.noise2 = random.choice(self.single_noises)
                        self.noise2 = self.noise2.lower().replace(" ", "_")

                self.walk = self.walk_dict[(self.noise1, self.noise2)]
                data_path = os.path.join(self.data_root, "n1_" + self.noise1 + "_n2_" + self.noise2)
                self.walk_datasets = path_to_dataset(self.walk, data_path)
                self.walk_ind = 0
            else:
                self.walk_ind += 1

        return sh_data.DataLoader(all_data, max_batch_size=None)

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = all_predicted_labels == np.array(self.ch_dataset.targets)

        return TaskResult(
            accuracy=accuracy, summary_metrics={Metric.Robustness: "accuracy"}
        )

def path_to_dataset(path, root):
    dir_list = []
    for i in range(len(path)):
        dir_list.append(os.path.join(root, "s1_" + str(float(path[i][0]) / 4) + "s2_" + str(float(path[i][1]) / 4)))
    return dir_list


def find_path(arr, target_val):
    cur_max = 99999999999
    cost_dict = {}
    path_dict = {}
    for i in range(1, arr.shape[0]):
        cost_dict, path_dict = traverse_graph(cost_dict, path_dict, arr, i, 0, target_val)

    for i in range(1, arr.shape[0]):
        cur_cost = abs(cost_dict[(i, 0)] / len(path_dict[(i, 0)]) - target_val)
        if cur_cost < cur_max:
            cur_max = cur_cost
            cur_path = path_dict[(i, 0)]

    return cur_path


def traverse_graph(cost_dict, path_dict, arr, i, j, target_val):
    if j >= arr.shape[1]:
        if (i,j) not in cost_dict.keys():
            cost_dict[(i,j)] = 9999999999999
            path_dict[(i,j)] = [9999999999999]
        return cost_dict, path_dict

    if i == 0:
        if (i,j) not in cost_dict.keys():
            cost_dict[(i,j)] = arr[i][j]
            path_dict[(i,j)] = [(i,j)]
        return cost_dict, path_dict


    if (i-1, j) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(cost_dict, path_dict, arr, i-1, j, target_val)
    if (i, j+1) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(cost_dict, path_dict, arr, i, j+1, target_val)

    if abs(((cost_dict[(i-1, j)] + arr[i][j]) / (len(path_dict[i-1, j]) + 1)) - target_val) < abs(((cost_dict[(i, j+1)] + arr[i][j]) / (len(path_dict[i, j+1]) + 1)) - target_val):
        cost_dict[(i, j)] = cost_dict[(i-1, j)] + arr[i][j]
        path_dict[(i, j)] = [(i,j)] + path_dict[(i-1, j)]
    else:
        cost_dict[(i, j)] = cost_dict[(i, j+1)] + arr[i][j]
        path_dict[(i, j)] = [(i,j)] + path_dict[(i, j+1)]

    return cost_dict, path_dict


if __name__ == "__main__":
    from shifthappens.models.torchvision import resnet18

    sh_benchmark.evaluate_model(resnet18(), "data")
