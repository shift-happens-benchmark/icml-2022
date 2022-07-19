import os
import io
import pandas as pd

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as tv_transforms
import torch.utils.data as data
import random
import pickle
import itertools

from shifthappens.tasks.ccc_imagenet_c import noise_transforms
from shifthappens.tasks.ccc_lmdb import ImageFolderLMDB, dset2lmdb


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




class WalkLoader(data.Dataset):
    """
        Generates a continuous walk through noises, at a desired baseline accuracy.

        Parameters
        ----------
        data_dir : str
            path to image dir (these are the images that the noises will be applied to)
        target_dir : str
            path to put generated files in
        seed: int
            seed for the random number generator
        frequency: int
            denotes how many images will be sampled from each subset
        base_amount: int
            this is about the entire size of the dataset (but not actually, because we need to start and end on the same noise)
        accuracy: int
            desired baseline accuracy to be used
        subset_size: int
            of the images in data_dir, how many should we use?
        Returns
        -------
        WalkLoader
            the generate function generates files, but if they already exist (or it finished generating files), returns a Dataset Object
    """
    def __init__(self, data_dir, target_dir, seed, frequency, base_amount, accuracy, subset_size):
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.seed = seed
        self.frequency = frequency
        self.base_amount = base_amount
        self.accuracy = accuracy
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

        pickle_path = os.path.join(self.target_dir, "ccc_accuracy_matrix.pickle")
        if not os.path.exists(pickle_path):
            url = "https://nc.mlcloud.uni-tuebingen.de/index.php/s/izTMnXkaHoNBZT4/download/ccc_accuracy_matrix.pickle"
            accuracy_matrix = pd.read_pickle(url)
            os.makedirs(self.target_dir, exist_ok=True)
            with open(pickle_path, 'wb') as f:
                pickle.dump(accuracy_matrix, f)
        else:
            with open(pickle_path, 'rb') as f:
                accuracy_matrix = pickle.load(f)


        noise_list = list(itertools.product(self.single_noises, self.single_noises))

        for i in range(len(noise_list)):
            noise1, noise2 = noise_list[i]
            if noise1 == noise2:
                continue
            current_accuracy_matrix = accuracy_matrix["n1_" + noise1 + "_n2_" + noise2]
            walk = find_path(current_accuracy_matrix, self.accuracy)

            accuracy_dict[(noise1, noise2)] = accuracy_matrix
            walk_dict[(noise1, noise2)] = walk

        keys = list(accuracy_dict.keys())
        cur_noises = random.choice(keys)

        walk = walk_dict[cur_noises]
        data_path = os.path.join(self.target_dir, "n1_" + noise1 + "_n2_" + noise2)
        walk_datasets = path_to_dataset(walk, data_path)

        self.walk_dict = walk_dict
        self.walk_ind = 0
        self.walk = walk
        self.walk_datasets = walk_datasets

        self.noise1 = random.choice(self.single_noises)
        self.first_noise1 = self.noise1
        self.noise2 = random.choice(self.single_noises)
        while self.noise1 == self.noise2:
            self.noise2 = random.choice(self.single_noises)


        self.lastrun = 0


    def generate_dataset(self):
        total_generated = 0
        all_data = None

        while True:
            temp_path = self.walk_datasets[self.walk_ind]
            severities = os.path.normpath(os.path.basename(temp_path)) # takes name of upper dir
            n1 = self.noise1
            n2 = self.noise2

            severities_split = severities.split('_')
            s1 = float(severities_split[1][:-2])
            s2 = float(severities_split[2])

            path = os.path.join(self.target_dir, 'n1_' + str(n1) + '_n2_' + str(n2)) + '_s1_' + str(s1) + '_s2_' + str(s2)
            if not os.path.exists(path):
                os.mkdir(path)

            if not (os.path.exists(os.path.join(path, 'lock.mdb')) and os.path.exists(os.path.join(path, 'data.mdb'))):
                generated_subset = ApplyTransforms(self.data_dir, n1, n2, s1, s2, self.subset_size)
                dset2lmdb(generated_subset, path)

            remainder = self.frequency
            while remainder > 0:
                cur_data = ImageFolderLMDB(db_path=path, transform=None)
                inds = np.random.permutation(len(cur_data))[:remainder]
                cur_data = torch.utils.data.Subset(cur_data, inds)
                remainder -= len(cur_data)

            if all_data is not None:
                all_data = torch.utils.data.ConcatDataset([all_data, cur_data])
            else:
                all_data = cur_data

            total_generated += self.frequency
            # print('total ', total_generated)
            if self.walk_ind == len(self.walk) - 1:
                self.noise1 = self.noise2

                if total_generated > self.base_amount and self.lastrun == 0:
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

                self.walk = self.walk_dict[(self.noise1, self.noise2)]
                data_path = os.path.join(self.target_dir, "n1_" + self.noise1 + "_n2_" + self.noise2)
                self.walk_datasets = path_to_dataset(self.walk, data_path)
                self.walk_ind = 0
            else:
                self.walk_ind += 1


class ApplyTransforms(data.Dataset):
    """
            Applies the desired noise transforms to a dataset. In our case, we apply 2 ImageNet-C noises at 2 severities

            Parameters
            ----------
            data_dir : str
                path to image dir (these are the images that the noises will be applied to)
            n1 : function
                noise function #1
            n2 : function
                noise function #2
            s1: int
                denotes the severity of noise #1
            s2: int
                denotes the severity of noise #2
            subset_size: int
                of the images in data_dir, how many should we use?
            Returns
            -------
            Dataset Object

        """
    def __init__(self, data_dir, n1, n2, s1, s2, subset_size):
        d = noise_transforms()
        self.data_root = data_dir
        self.n1 = d[n1]
        self.n2 = d[n2]
        self.s1 = s1
        self.s2 = s2

        self.trn = tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224)])
        all_paths = []

        for path, dirs, files in os.walk(self.data_dir):
            for name in files:
                all_paths.append(os.path.join(path, name))

        np.random.shuffle(all_paths)
        self.paths = all_paths
        self.paths = self.paths[:subset_size]
        all_classes = os.listdir(os.path.join(self.data_dir))

        target_list = []
        for cur_path in self.paths:
            cur_class = os.path.normpath(os.path.basename(os.path.abspath(os.path.join(cur_path, os.pardir)))) # takes name of parent dir
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


