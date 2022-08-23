"""
This module provides functionality to access the predictions of a model for the ImageNet validation set.
Further, the predictions can be cached and loaded from the cache to reduce computational costs.
Note, that for this to work :py:data:shifthappens.data.imagenet.ImageNetValidationData
must be set to the ImageNet validation set path. path.
"""
import os
import shutil

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.torch as sh_data_torch
from shifthappens.data.base import DataLoader

#: Must be set to ImageNet validation set path (either absolute or relative to working directory).
ImageNetValidationData = "../../../../../imagenet"

#: Must be set to models' results caching directory (either absolute or relative to working directory). If the folder does not exist, it will be created.
ImageNetValidationPredictionsCache = "shifthappens/cache"


def _check_imagenet_folder():
    """
    Checks if path to the ImageNet validation set folder is defined, the folder
    exists and contains a thousand folders (one per class).
    """
    assert (
        ImageNetValidationData is not None
    ), "ImagenetValidationData path is not specified."
    assert os.path.exists(ImageNetValidationData), (
        "You have specified an incorrect path to the ImageNet validation set. "
        "Files not found at location specified in shithappens.data.imagenet.ImageNetValidationData."
    )

    assert (
        len(os.listdir(ImageNetValidationData)) == 1000
    ), "ImageNetValidationData folder contains less or more folders than ImageNet classes."


def get_imagenet_validation_loader(max_batch_size=128) -> DataLoader:
    """
    Creates a :py:class:`shifthappens.data.base.DataLoader` for the validation set of ImageNet.
    Note that the path to ImageNet validation set
    :py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be specified.

    Args:
        max_batch_size: How many samples allowed per batch to load.

    Returns:
        ImageNet validation set data loader.
    """
    _check_imagenet_folder()
    transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    imagenet_val_dataset = sh_data_torch.IndexedTorchDataset(
        sh_data_torch.ImagesOnlyTorchDataset(
            tv_datasets.ImageFolder(root=ImageNetValidationData, transform=transform)
        )
    )

    imagenet_val_dataloader = DataLoader(
        imagenet_val_dataset, max_batch_size=max_batch_size
    )

    return imagenet_val_dataloader


def get_cached_predictions(cls) -> dict:
    """
    Checks whether there exist cached results for the model's class and if so, returns them.
    Note that the path to ImageNet validation set
    :py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be specified.

    Args:
        cls: Model's class. Used for specifying folder name.

    Returns:
        Dictionary of loaded model predictions on ImageNet validation set.
    """
    assert (
        ImageNetValidationPredictionsCache is not None
    ), "Cannot get cached model results. ImageNetValidationPredictionsCache path is not specified."

    load_path = os.path.join(
        ImageNetValidationPredictionsCache, cls.__class__.__name__, ""
    )

    assert os.path.exists(
        load_path
    ), f"Cannot get cached model results. {load_path} folder not found."
    assert (
        len(os.listdir(load_path)) != 0
    ), f"Cannot get cached model results. {load_path} folder is empty."

    result_dict = dict()
    for file in os.listdir(load_path):
        result = np.load(load_path + file)
        result_dict[file.rstrip(".npy")] = result
    return result_dict


def cache_predictions(cls, imagenet_validation_result):
    """
    Caches model predictions in cls-named folder and load
    model predictions from it. Note that the path to ImageNet validation set
    :py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be specified as
    well as :py:data:`shifthappens.data.imagenet.ImageNetValidationPredictionsCache`.

    Args:
        cls: Model's class. Used for specifying folder name.
        imagenet_validation_result (ModelResult): Model's prediction on ImageNet
            validation set.
    """

    assert (
        ImageNetValidationPredictionsCache is not None
    ), "Cannot cache model results. ImageNetValidationPredictionsCache path is not specified."
    save_path = os.path.join(
        ImageNetValidationPredictionsCache, cls.__class__.__name__, ""
    )

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for result_type in imagenet_validation_result.__slots__:
        result = getattr(imagenet_validation_result, str(result_type))
        if result is not None:
            np.save(save_path + str(result_type), result)


def is_cached(cls) -> bool:
    """
    Checks if model's results are cached in cls-named folder. Note that the path to the ImageNet validation set
    :py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be specified as
    well as :py:data:`shifthappens.data.imagenet.ImageNetValidationPredictionsCache`.

    Args:
        cls: Model's class. Used for specifying folder name.

    Returns:
        ``True`` if model's results are cached, ``False`` otherwise.
    """
    assert (
        ImageNetValidationPredictionsCache is not None
    ), "Cannot find cached model results. ImageNetValidationPredictionsCache path is not specified."
    load_path = os.path.join(
        ImageNetValidationPredictionsCache, cls.__class__.__name__, ""
    )

    try:
        cached_files = os.listdir(load_path)
    except FileNotFoundError:
        print(f"There is no cached model results on ImageNet at {load_path}.")
        return False
    return True


def load_imagenet_targets() -> np.ndarray:
    """
    Returns the ground-truth labels of the ImageNet valdation set.
    """
    _check_imagenet_folder()
    return tv_datasets.ImageFolder(root=ImageNetValidationData).targets
