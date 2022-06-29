"""
Implementation of CCC: Continuously Changing Corruptions
Note: This Task only implements the data reading portion of the dataset.
In addition to this file, we submitted a file used to generate the data itself.
"""

import dataclasses
import os
import random
import itertools
import pickle
import torch

import numpy as np
import torchvision.transforms as tv_transforms
from PIL import Image
import io

import shifthappens.data.base as sh_data
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.base import parameter

from shifthappens.tasks.ccc_imagenet_c import noise_transforms
from shifthappens.tasks.ccc_utils import ImageFolderLMDB, dset2lmdb


@sh_benchmark.register_task(
    name="CCC", relative_data_folder="ccc", standalone=True
)
@dataclasses.dataclass
class CCC(Task):
    seed: int = parameter(
        description="random seed used in the dataset building process",
    )
    frequency: int = parameter(
        description="represents how many images are sampled from each subset",
    )
    base_amount: int = parameter(
        description="represents how large the base dataset is",
    )
    accuracy: int = parameter(
        description="represents the baseline accuracy of walk",
    )

    def setup(self):
        self.loader = WalkLoader(os.path.join(self.data_root, "ccc"), os.path.join(self.data_root, "ccc", "accuracies"),
                                 self.seed, self.frequency, self.base_amount, self.accuracy, 20000)

    def _prepare_dataloader(self) -> DataLoader:
        self.setup()
        data = self.loader.get()
        return sh_data.DataLoader(data, max_batch_size=None)

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
        if (i, j) not in cost_dict.keys():
            cost_dict[(i, j)] = 9999999999999
            path_dict[(i, j)] = [9999999999999]
        return cost_dict, path_dict

    if i == 0:
        if (i, j) not in cost_dict.keys():
            cost_dict[(i, j)] = arr[i][j]
            path_dict[(i, j)] = [(i, j)]
        return cost_dict, path_dict

    if (i - 1, j) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(cost_dict, path_dict, arr, i - 1, j, target_val)
    if (i, j + 1) not in cost_dict.keys():
        cost_dict, path_dict = traverse_graph(cost_dict, path_dict, arr, i, j + 1, target_val)

    if abs(((cost_dict[(i - 1, j)] + arr[i][j]) / (len(path_dict[i - 1, j]) + 1)) - target_val) < abs(
            ((cost_dict[(i, j + 1)] + arr[i][j]) / (len(path_dict[i, j + 1]) + 1)) - target_val):
        cost_dict[(i, j)] = cost_dict[(i - 1, j)] + arr[i][j]
        path_dict[(i, j)] = [(i, j)] + path_dict[(i - 1, j)]
    else:
        cost_dict[(i, j)] = cost_dict[(i, j + 1)] + arr[i][j]
        path_dict[(i, j)] = [(i, j)] + path_dict[(i, j + 1)]

    return cost_dict, path_dict


class WalkLoader(tv_transforms.Dataset):
    def __init__(self, data_root, accuracies_folder, seed, frequency, base_amount, accuracy, subset_size):
        self.data_root = data_root
        self.accuracies_folder = accuracies_folder
        self.seed = seed
        self.frequency = frequency
        self.accuracy = accuracy
        self.base_amount = base_amount
        self.subset_size = subset_size

        random.seed(self.seed)
        np.random.seed(self.seed)
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
            # 'brightness', # these noises aren't used for baseline accuracy=20
            'contrast',
            'elastic',
            'pixelate',
            # 'jpeg' # these noises aren't used for baseline accuracy=20
        ]

        noise_list = list(itertools.product(self.single_noises, self.single_noises))

        for i in range(len(noise_list)):
            noise1, noise2 = noise_list[i]
            if noise1 == noise2:
                continue
            noise1 = noise1.lower().replace(" ", "_")
            noise2 = noise2.lower().replace(" ", "_")

            results_path = os.path.join(self.accuracies_folder, "n1_" + noise1 + "_n2_" + noise2, "accuracies.pkl")

            with open(results_path, 'rb') as f:
                accuracy_matrix = pickle.load(f)

            walk = find_path(accuracy_matrix, self.accuracy)

            accuracy_dict[(noise1, noise2)] = accuracy_matrix
            walk_dict[(noise1, noise2)] = walk

        keys = list(accuracy_dict.keys())
        cur_noises = random.choice(keys)
        noise1 = cur_noises[0].lower().replace(" ", "_")
        noise2 = cur_noises[1].lower().replace(" ", "_")

        walk = walk_dict[cur_noises]
        data_path = os.path.join(self.data_root, "n1_" + noise1 + "_n2_" + noise2)
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

    def get(self):
        total = 0
        all_data = None

        while True:
            path_split = self.walk_datasets[self.walk_ind].split('/')
            noise_names = path_split[-2]
            severities = path_split[-1]

            noises_split = noise_names.split('_')
            n1 = noises_split[1]
            n2 = noises_split[3]

            severities_split = severities.split('_')
            s1 = float(severities_split[1][:-2])
            s2 = float(severities_split[2])

            path = os.path.join(self.output_path, 'n1_' + str(n1) + '_n2_' + str(n2)) + '_s1_' + str(s1) + '_s2_' + str(
                s2)
            if not os.path.exists(path):
                os.mkdir(path)
                cur_data = ApplyTransforms(self.data_root, n1, n2, s1, s2, self.subset_size)
                dset2lmdb(cur_data, path)

            remainder = self.frequency
            while remainder > 0:
                cur_data = ImageFolderLMDB(db_path=path, transform=self.transform)
                inds = np.random.permutation(len(cur_data))[:remainder]
                cur_data = torch.utils.data.Subset(cur_data, inds)
                remainder -= self.freq

            if all_data is not None:
                all_data = torch.utils.data.ConcatDataset([all_data, cur_data])
            else:
                all_data = cur_data

            total += self.frequency
            self.lifetime_total += self.frequency
            if self.walk_ind == len(self.walk) - 1:
                self.noise1 = self.noise2

                if self.lifetime_total > self.base_amount and self.lastrun == 0:
                    if self.noise1 != self.first_noise1:
                        self.noise2 = self.first_noise1
                        self.lastrun = 1
                    else:
                        return all_data
                elif self.lastrun == 1:
                    return all_data
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


class ApplyTransforms(tv_transforms.Dataset):
    def __init__(self, data_root, n1, n2, s1, s2, frequency):
        d = noise_transforms()
        self.data_root = data_root
        self.n1 = d[n1]
        self.n2 = d[n2]
        self.s1 = s1
        self.s2 = s2

        self.trn = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224)])
        all_paths = []

        for path, dirs, files in os.walk(self.data_root):
            for name in files:
                all_paths.append(os.path.join(path, name))

        np.random.shuffle(all_paths)
        self.paths = all_paths
        self.paths = self.paths[:frequency]
        all_classes = os.listdir(os.path.join(self.data_root))

        target_list = []
        for cur_path in self.paths:
            cur_class = cur_path.split('/')[-2]
            cur_class = all_classes.index(cur_class)
            target_list.append(cur_class)

        self.targets = target_list

    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]

        with open(path, 'rb') as f:
            img = Image.open(f)

        img = img.convert('RGB')
        img = self.trn(img)

        if self.s1 > 0:
            img = self.n1(img, self.s1)
            img = Image.fromarray(np.uint8(img))
        if self.s2 > 0:
            img = self.n2(img, self.s2)

        if self.s2 > 0:
            img = Image.fromarray(np.uint8(img))
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85, optimize=True)
        corrupted_img = output.getvalue()
        return corrupted_img, target

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet50
    sh_benchmark.evaluate_model(ResNet50(), "data")
