"""Example for a Shift Happens task: ImageNet-R"""

import dataclasses
import os
import itertools
import random

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
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="ImageNet-R", relative_data_folder="imagenet_r", standalone=True
)
@dataclasses.dataclass
class ImageNetR(Task):
    resources = [
        (
            "imagenet-r.tar",
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            "A61312130A589D0CA1A8FCA1F2BD3337",
        )
    ]

    def setup(self):
        dataset_folder = os.path.join(self.data_root, "imagenet-r")
        if not os.path.exists(dataset_folder):
            # download data
            for file_name, url, md5 in self.resources:
                sh_utils.download_and_extract_archive(
                    url, self.data_root, md5, file_name
                )

        test_transform = tv_transforms.Compose(
            [
                tv_transforms.ToTensor(),
                tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
            ]
        )

        self.ch_dataset = tv_datasets.ImageFolder(
            root=dataset_folder, transform=test_transform
        )
        self.images_only_dataset = sh_data_torch.IndexedTorchDataset(
            sh_data_torch.ImagesOnlyTorchDataset(self.ch_dataset)
        )

    def _prepare_dataloader(self) -> DataLoader:
        return sh_data.DataLoader(self.images_only_dataset, max_batch_size=None)

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


class CottaRecorder(data.Dataset):
    def __init__(self, data_root, results_root, freq, seed, cutoff, cost=20, transform=None,
                 loader=default_loader, skip_inds=[]):
        self.data_root = data_root
        self.results_root = results_root
        self.loader = loader
        self.transform = transform

        self.freq = freq
        self.seed = seed
        self.cutoff = cutoff
        self.cost = cost
        self.skip_inds = skip_inds

        accuracy_dict = {}
        walk_dict = {}
        self.single_noises = ['gaussian_noise',
                              'shot_noise',
                              'impulse_noise',
                              'defocus_blur',
                              'glass_blur',
                              'motion_blur',
                              'zoom_blur',
                              'snow',
                              'frost',
                              'fog',
                              # 'brightness',
                              'contrast',
                              'elastic',
                              'pixelate',
                              # 'jpeg'
                              ]

        noise_list = list(itertools.product(self.single_noises, self.single_noises))

        for i in range(len(noise_list)):
            noise1, noise2 = noise_list[i]
            if noise1 == noise2:
                continue
            noise1 = noise1.lower().replace(" ", "_")
            noise2 = noise2.lower().replace(" ", "_")

            results_path = os.path.join(self.results_root, "n1_" + noise1 + "_n2_" + noise2, "accuracies.txt")
            accuracy_matrix = collect(results_path)

            walk = find_path(accuracy_matrix, self.cost)

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

        while True:
            path_split = self.walk_datasets[self.walk_ind].split('/')
            noise_names = path_split[-2]
            severities = path_split[-1]

            noise_names = noise_names.replace('_noise', '')
            noise_names = noise_names.replace('_blur', '')

            noises_split = noise_names.split('_')
            n1 = noises_split[1]
            n2 = noises_split[3]

            severities_split = severities.split('_')
            s1 = float(severities_split[1][:-2])
            s2 = float(severities_split[2])

            pathqb = os.path.join('/mnt/qb/work/bethge/opress/50k_noises',
                                  'n1_' + str(n1) + '_n2_' + str(n2)) + '_s1_' + str(s1) + '_s2_' + str(s2)
            if not os.path.exists(pathqb):
                os.mkdir(pathqb)
                cur_data = Loader3Transforms(self.data_root, n1, n2, s1, s2, 20000)
                if pathqb == '/mnt/qb/work/bethge/opress/50k_noises/n1_elastic_n2_shot_s1_0.25_s2_3.0':
                    print('cur data len ', len(cur_data))
                dset2lmdb(cur_data, pathqb)


            total += self.freq
            self.lifetime_total += self.freq
            if self.walk_ind == len(self.walk) - 1:
                self.noise1 = self.noise2

                if self.lifetime_total > 750000 and self.lastrun == 0:
                    if self.noise1 != self.first_noise1:
                        self.noise2 = self.first_noise1
                        self.lastrun = 1
                    else:
                        return
                elif self.lastrun == 1:
                    return
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


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cpu", max_batch_size=128), "data")
