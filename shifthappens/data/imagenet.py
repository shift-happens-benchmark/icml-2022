"""
This module provides functionality to access ImageNet validation set ground truth targets,
load and cache results of model's prediction on ImageNet validation set. Note,
that every model inherited from :py:class:`Model <shifthappens.models.base.Model>`
implicitly evaluates on ImageNet validation set and caches the results, thus
:py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be set to the ImageNet validation set
path.
"""
import os
import shutil

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.torch as sh_data_torch
from shifthappens.data.base import DataLoader

#: Must be set to ImageNet validation set PATH.
ImageNetValidationData = "shifthappens/imagenet"

#: Must be set to models' results caching directory.
ImageNetValidationPredictionsCache = "shifthappens/cache"


def _check_imagenet_folder():
    """
    Checks if path to ImageNet folder is defined, the folder exists and contains
    a thousand folders.
    """
    assert (
        ImageNetValidationData is not None
    ), "ImagenetValidationData path is not specified."
    assert os.path.exists(
        ImageNetValidationData
    ), "You have specified incorrect path to ImageNet. Files not found at the ImageNetValidationData path."

    assert (
        len(os.listdir(ImageNetValidationData)) == 1000
    ), "ImageNetValidationData folder contains less or more folders then ImageNet classes."


def get_imagenet_val_dataloader(max_batch_size=128) -> DataLoader:
    """
    Returns ImageNet validation set DataLoader. Note
    that path to ImageNet validation set :py:data:`shifthappens.data.imagenet.ImageNetValidationData`
    must be specified.

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
    Looks up in cache directory for a cls-named folder and load
    model predictions from it. Note that path to ImageNet validation set
    :py:data:`shifthappens.data.imagenet.ImageNetValidationData` must be specified.

    Args:
        cls: Model's class. Used for specifying folder name.

    Returns:
        Dictionary of loaded model predictions on ImageNet validation set.
    """
    assert (
        ImageNetValidationPredictionsCache is not None
    ), "Cannot get cached model results. ImageNetValidationPredictionsCache path is not specified."
    load_path = ImageNetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"
    result_dict = dict()
    for file in os.listdir(load_path):
        result = np.load(load_path + file)
        result_dict[file.rstrip(".npy")] = result
    return result_dict


def cache_predictions(cls, imagenet_validation_result):
    """
    Caches model predictions in cls-named folder and load
    model predictions from it. Note that path to ImageNet validation set
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
    save_path = ImageNetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for result_type in imagenet_validation_result.__slots__:
        result = getattr(imagenet_validation_result, str(result_type))
        if result is not None:
            np.save(save_path + str(result_type), result)


def is_cached(cls) -> bool:
    """
    Checks if model's results are cached in cls-named folder. Note
    that path to ImageNet validation set
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
    load_path = ImageNetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"

    try:
        cached_files = os.listdir(load_path)
    except FileNotFoundError:
        print(f"There is no cached model results on ImageNet at {load_path}.")
        return False
    return True


def load_imagenet_targets() -> np.ndarray:
    """
    Returns ground truth targets of ImageNet validations set.
    """
    _check_imagenet_folder()
    return tv_datasets.ImageFolder(root=ImageNetValidationData).targets
