import os
import shutil

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.torch as sh_data_torch
from shifthappens.data.base import DataLoader

ImagenetValidationData = "../imagenet"
ImagenetValidationPredictionsCache = "../cache"


def get_imagenet_val_dataloader(max_batch_size=128) -> DataLoader:
    """
    A getter function which returns ImageNet validation set DataLoader. Note
    that path to ImageNet validation set ``ImagenetValidationData`` need to be specified.
    Args:
        max_batch_size: How many samples allowed per batch to load.

    Returns:
    ImageNet validation set DataLoader.
    """
    assert (
        ImagenetValidationData is not None
    ), "ImagenetValidationData path is not specified"

    transform = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        ]
    )
    imagenet_val_dataset = sh_data_torch.IndexedTorchDataset(
        sh_data_torch.ImagesOnlyTorchDataset(
            tv_datasets.ImageFolder(root=ImagenetValidationData, transform=transform)
        )
    )

    imagenet_val_dataloader = DataLoader(
        imagenet_val_dataset, max_batch_size=max_batch_size
    )

    return imagenet_val_dataloader


def get_cached_predictions(cls) -> dict:
    """
    This function look up in cache directory for a cls-named folder and load
    model predictions from it. Note that path to ImageNet validation set
    ``ImagenetValidationData`` need to be specified.
    Args:
        cls: Model's class. Used for specifying folder name.

    Returns:
    Dictionary of loaded model predictions on ImageNet validation set.
    """
    assert (
        ImagenetValidationPredictionsCache is not None
    ), "ImagenetValidationPredictionsCache path is not specified"
    load_path = ImagenetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"
    result_dict = dict()
    for file in os.listdir(load_path):
        result = np.load(load_path + file)
        assert (
            len(result) == 50000
        ), f"{cls.__class__.__name__} result on ImageNet is corrupted: {load_path + file} has length {len(result)}"
        result_dict[file.rstrip(".npy")] = result
    return result_dict


def cache_predictions(cls, imagenet_validation_result):
    """
    This function caches model predictions in cls-named folder and load
    model predictions from it. Note that path to ImageNet validation set
    ``ImagenetValidationData`` need to be specified.
    Args:
        cls: Model's class. Used for specifying folder name.
        imagenet_validation_result (ModelResult): Model's prediction on ImageNet
        validation set.
    """

    assert (
        ImagenetValidationPredictionsCache is not None
    ), "ImagenetValidationPredictionsCache path is not specified"
    save_path = ImagenetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for result_type in imagenet_validation_result.__slots__:
        result = getattr(imagenet_validation_result, str(result_type))
        if result is not None:
            np.save(save_path + str(result_type), result)


def is_cached(cls) -> bool:
    """
    This function checks if model's results are cached in cls-named folder. Note
    that path to ImageNet validation set ``ImagenetValidationData`` need to be specified.
    Args:
        cls: Model's class. Used for specifying folder name.

    Returns:
    True if model's results are cached, False otherwise.
    """
    assert (
        ImagenetValidationPredictionsCache is not None
    ), "ImagenetValidationPredictionsCache path is not specified"
    load_path = ImagenetValidationPredictionsCache + "/" + cls.__class__.__name__ + "/"

    try:
        cached_files = os.listdir(load_path)
    except FileNotFoundError:
        print(f"There is no cached model results on Imagenet in {load_path} folder.")
        return False
    return True


def load_imagenet_targets() -> np.ndarray:
    """This function returns grounf truth targets of ImageNet validations set."""
    assert (
        ImagenetValidationData is not None
    ), "ImagenetValidationData path is not specified"

    return tv_datasets.ImageFolder(root=ImagenetValidationData).targets


ImagenetTargets = load_imagenet_targets()
