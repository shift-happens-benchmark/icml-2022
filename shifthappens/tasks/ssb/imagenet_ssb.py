import numpy as np
import os
import pickle
from copy import deepcopy

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ImageNetSubsetDataset(Dataset):

    def __init__(self, root, classes_to_keep, transform):

        self.root = root
        self.classes = classes_to_keep
        self.transform = transform

        # List all samples
        root_dirs = [os.path.join(root, c) for c in self.classes]

        samples = []
        labels = []

        for r, c in zip(root_dirs, self.classes):
            files = os.listdir(r)
            samples.extend([os.path.join(r, f) for f in files])
            labels.extend([c] * len(files))

        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):

        path = self.samples[item]
        image = default_loader(path)
        image = self.transform(image).float()

        cls = self.labels[item]

        return image, cls

def pad_to_longest(list1, list2):

    if len(list2) > len(list1):

        list1 = [None] * (len(list2) - len(list1)) + list1

    elif len(list1) > len(list2):

        list2 = [None] * (len(list1) - len(list2)) + list2

    else:

        pass

    return list1, list2


def get_imagenet_osr_class_splits(imagenet21k_class_to_idx, precomputed_split_dir, osr_split='Easy'):

    split_to_key = {
        'Easy': 'easy_i21k_classes',
        'Hard': 'hard_i21k_classes'
    }

    # Load splits
    with open(precomputed_split_dir, 'rb') as handle:
        precomputed_info = pickle.load(handle)

    osr_wnids = precomputed_info[split_to_key[osr_split]]
    selected_osr_classes_class_indices = \
        [imagenet21k_class_to_idx[cls_name] for cls_name in osr_wnids]

    return selected_osr_classes_class_indices


def subsample_dataset(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_split(train_dataset, val_split=0.2):

    val_dataset = deepcopy(train_dataset)
    train_dataset = deepcopy(train_dataset)

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    # Get training/validation datasets based on selected idxs
    train_dataset = subsample_dataset(train_dataset, train_idxs)
    val_dataset = subsample_dataset(val_dataset, val_idxs)

    return train_dataset, val_dataset


def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def get_imagenet_ssb_datasets(test_transform, imagenet21k_root, osr_split_path):

    print('Loading ImageNet21K SSB Val Sets...')

    # Load splits
    with open(osr_split_path, 'rb') as handle:
        precomputed_info = pickle.load(handle)

    easy_wnids = precomputed_info['easy_i21k_classes']
    hard_wnids = precomputed_info['hard_i21k_classes']

    easy_ssb_set = ImageNetSubsetDataset(root=os.path.join(imagenet21k_root, 'val'), classes_to_keep=easy_wnids,
                                         transform=test_transform)
    hard_ssb_set = ImageNetSubsetDataset(root=os.path.join(imagenet21k_root, 'val'), classes_to_keep=hard_wnids,
                                         transform=test_transform)

    return easy_ssb_set, hard_ssb_set

sh_benchmark.register_task(
    name="SSB", relative_data_folder="imagenet21k_p", standalone=True
)