"""Some utils for extracting ObjectNet and mapping the classes."""
import os
import time
import urllib.error
import zipfile
from typing import Optional

import numpy as np
import pandas as pd
from torchvision import datasets


# updated from https://github.com/pytorch/vision/blob0cba9b7845795d6be7b164037461ea6e9265f6a2/torchvision/datasets/utils.py
def download_and_extract_zip_with_pwd(
    url: str,
    data_folder: str,
    md5: str,
    filename: Optional[str],
    password: Optional[str],
    remove_finished: bool = False,
    n_retries: int = 3,
) -> None:
    """
    Downloads and extracts and archive using torchvision.

    Args:
        url (str): URL to download.
        data_folder (str): Where to save the downloaded file.
        md5 (str): MD5 hash of the archive.
        filename (str, optional): Name under which the archive will be saved locally.
        password (str, optional): Archive's password.
        remove_finished (bool, optional): Remove archive after extraction?
        n_retries (int): How often to retry the download in case of connectivity issues.
    """

    if not filename:
        filename = os.path.basename(url)

    import torchvision.datasets.utils as tv_utils

    for _ in range(n_retries):
        try:
            tv_utils.download_url(url, data_folder, filename, md5)
            break
        except urllib.error.URLError:
            print(f"Download of {url} failed; wait 5s and then try again.")
            time.sleep(5)

    archive = os.path.join(data_folder, filename)
    print(f"Extracting {archive} to {data_folder}")
    with zipfile.ZipFile(archive, "r") as zip_ref:
        if password is not None:
            zip_ref.extractall(data_folder, pwd=bytes(password, "utf-8"))
        else:
            zip_ref.extractall(data_folder)
    if remove_finished:
        os.remove(archive)


def folder_map(mapping_path: str) -> dict:
    """
    Creates mapping folder name -> ObjectNet class -> ImageNet class.
    Args:
        mapping_path (str): Path to the mappings folder.
    Returns:
        dict: Mapping from folder name to ImageNet class.
    """

    class_to_imagenet = pd.read_json(
        os.path.join(mapping_path, "objectnet_to_imagenet_1k.json"), typ="series"
    ).to_dict()
    class_to_folder_name = pd.read_json(
        os.path.join(mapping_path, "folder_to_objectnet_label.json"), typ="series"
    ).to_dict()
    mapping = {
        k: class_to_imagenet[v]
        for k, v in class_to_folder_name.items()
        if v in class_to_imagenet
    }
    return mapping


class ImageFolderImageNetClassesIntersection(datasets.ImageFolder):
    """
    This is required for handling empty folders that cannot be mapped to
    classes in ImageNet.
    """

    def __init__(self, mapping_path, *args, **kwargs):
        self.folder_map = folder_map(mapping_path)
        super().__init__(*args, **kwargs)

    def find_classes(self, directory):
        """Rejects the folders that cannot be mapped to classes in ImageNet."""
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {
            cls_name: i
            for i, cls_name in enumerate(classes)
            if cls_name in self.folder_map
        }
        return classes, class_to_idx


def class_intersection_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    imagenet_id_to_objectnet_id: dict,
) -> float:
    """
    Computes the accuracy of the predictions on the intersection of classes in ImageNet and ObjectNet.
    Args:
        targets: Target labels.
        predictions: Predicted labels.
        imagenet_id_to_objectnet_id: Mapping from ImageNet class indices to ObjectNet class indices.

    Returns:
        float: Accuracy of the predictions on the intersection of classes in ImageNet and ObjectNet.
    """

    assert targets.shape == predictions.shape
    imagenet_id_to_objectnet_id = {
        int(k): v for k, v in imagenet_id_to_objectnet_id.items()
    }
    result = []
    for i in range(len(predictions)):
        if predictions[i] in imagenet_id_to_objectnet_id:
            predictions[i] = imagenet_id_to_objectnet_id[predictions[i]]
        else:
            predictions[i] = -1

        result.append(targets[i] == predictions[i])
    if len(result) == 0:
        return 0.0
    else:
        return sum(result) / len(result)
