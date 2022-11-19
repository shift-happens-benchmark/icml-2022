"""Subsets of ImageNet21k."""

import os

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

import shifthappens.config


class ImageNetSubsetDataset(Dataset):
    """
    This wrapper is required for loading correct classes for every subset.
    """

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


def _get_imagenet_ssb_subset(test_transform, imagenet21k_root, osr_split, subset_type):
    """Get dataset of subset of ImageNet21k with chosen transformation.
    
    Args:
        test_transform: Transformation.
        imagenet21k_root: Path to root folder of ImageNet21k dataset.
        osr_split: Name of split.
        subset_type: Type of subset. Either easy or hard.
    
    Returns:
        Dataset.
    """
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
        f"ImageNet-21K-P (winter 21 version) data not downloaded not found in {shifthappens.config.imagenet21k_preprocessed_validation_path} "
        f"\n Please download processed data according to:"
        "https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md"
        f"\n Or download raw data and processes it according to:"
        "https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_script.sh"
    )
