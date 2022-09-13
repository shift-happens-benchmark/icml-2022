import os
import pickle
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import shifthappens.config


class ImageNetSubsetDataset(Dataset):
    def __init__(self, root, classes_to_keep, transform):

        self.root = root
        self.classes = classes_to_keep
        self.transform = transform

        # List all samples
        root_dirs = [os.path.join(root, c) for c in self.classes]

        samples = ["image not found"] * len(root_dirs) * 50
        labels = [-1] * len(root_dirs) * 50
        for i, r_c_pair in enumerate(zip(root_dirs, self.classes)):
            r, c = r_c_pair
            files = os.listdir(r)
            for j, file in enumerate(files):
                samples[i * 50 + j] = os.path.join(r, file)
                labels[i * 50 + j] = c

        assert "image not found" not in samples, (
            "Seems like ImageNet21K-P validation set is corrupted and some class is missing images."
            "Please check if every class contains exactly 50 images. "
        )
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


def get_imagenet_osr_class_splits(
    imagenet21k_class_to_idx, precomputed_split_dir, osr_split="Easy"
):

    split_to_key = {"Easy": "easy_i21k_classes", "Hard": "hard_i21k_classes"}

    # Load splits
    with open(precomputed_split_dir, "rb") as handle:
        precomputed_info = pickle.load(handle)

    osr_wnids = precomputed_info[split_to_key[osr_split]]
    selected_osr_classes_class_indices = [
        imagenet21k_class_to_idx[cls_name] for cls_name in osr_wnids
    ]

    return selected_osr_classes_class_indices


def subsample_dataset(dataset, idxs):

    dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]
    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(1000)):

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

        v_ = np.random.choice(
            cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),)
        )
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

        rand_idxs = np.random.choice(
            range(len(dataset1)),
            size=(
                len(
                    dataset2,
                )
            ),
        )
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(
            range(len(dataset2)),
            size=(
                len(
                    dataset1,
                )
            ),
        )
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2


def _get_imagenet_ssb_subset(test_transform, imagenet21k_root, osr_split, subset_type):
    assert subset_type in ["easy", "hard"], "subset_type must be 'easy' or 'hard'"
    print(f"Loading ImageNet21K SSB val {subset_type} set...")

    wnids = osr_split[f"{subset_type}_i21k_classes"]

    ssb_set = ImageNetSubsetDataset(
        root=imagenet21k_root,
        classes_to_keep=wnids,
        transform=test_transform,
    )

    return ssb_set


def assert_data_downloaded(osr_split: dict, imagenet21k_root: str):
    """
    Check if data is downloaded and contains necessary folders.

        Args:
            osr_split: osr_split dict
            imagenet21k_root: path to imagenet21k-p validations
                set specified in shifthappens.config.imagenet21k_preprocessed_validation_path

    """
    easy_wnids = osr_split["easy_i21k_classes"]
    hard_wnids = osr_split["hard_i21k_classes"]
    all_wnids = easy_wnids.tolist() + hard_wnids

    assert os.path.exists(
        shifthappens.config.imagenet21k_preprocessed_validation_path
    ), (
        "You have specified an incorrect path to the ImageNet21K-P validation set. "
        "Files not found at location specified in shifthappens.config.imagenet21k_preprocessed_validation_path."
    )

    paths = os.listdir(imagenet21k_root)
    assert all([x in paths for x in all_wnids]), (
        f"ImageNet-21K-P data not downloaded not found in {shifthappens.config.imagenet21k_preprocessed_validation_path} "
        f"\n Please download processed data according to:"
        "https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md"
        f"\n Or download raw data and processes it according to:"
        "https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh"
    )
