"""A task for evaluating the classification accuracy on ImageNet-C.

While the dataset is ImageNet-C the task's definition is a bit different than the usual evaluation
paradigm: we allow the model to access (1) the unlabeled test set separately for every corruption
and (2) allow it to make it's prediction based on a batch of samples coming from the same
corruption type.
"""

import dataclasses
import os
import typing
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import torchvision.datasets as tv_datasets
import torchvision.transforms as tv_transforms

import shifthappens.data.base as sh_data
import shifthappens.data.torch as sh_data_torch
import shifthappens.task_data.task_metadata
import shifthappens.utils as sh_utils
from shifthappens import benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.base import PredictionTargets
from shifthappens.tasks.base import abstract_variable
from shifthappens.tasks.base import parameter
from shifthappens.tasks.base import Task
from shifthappens.tasks.base import variable
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@dataclasses.dataclass
class ImageNetSingleCorruptionTypeBase(Task):
    """Evaluate the classification accuracy on a single corruption type
    of the ImageNet-C dataset [1]. Each corruption type has 5 different
    severity levels. The raw images (before corruptions) in this dataset
    come from the validation set of ImageNet.

    [1] Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
        Dan Hendrycks and Thomas Dietterich. 2019.
    """

    resource: Tuple[str, ...] = abstract_variable()

    severity: int = parameter(
        default=1,
        options=(1, 2, 3, 4, 5),
        description="severity of the image corruption",
    )

    max_batch_size: Optional[int] = None

    def setup(self):
        """Load and prepare data."""

        folder_name, archive_name, url, md5 = self.resource

        dataset_folder = os.path.join(self.data_root, folder_name, str(self.severity))
        if not os.path.exists(dataset_folder):
            # download data
            sh_utils.download_and_extract_archive(
                url, self.data_root, md5, archive_name
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
        return sh_data.DataLoader(
            self.images_only_dataset, max_batch_size=self.max_batch_size
        )

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        dataloader = self._prepare_dataloader()

        all_predicted_labels_list = []
        for predictions in model.predict(
            dataloader, PredictionTargets(class_labels=True)
        ):
            all_predicted_labels_list.append(predictions.class_labels)
        all_predicted_labels = np.concatenate(all_predicted_labels_list, 0)

        accuracy = (
            all_predicted_labels
            == np.array(self.ch_dataset.targets)[: len(all_predicted_labels)]
        )

        return TaskResult(
            accuracy=accuracy,
            mce=1.0 - accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


# noise corruptions
@sh_benchmark.register_task(
    name="ImageNet-C (Gaussian Noise)",
    relative_data_folder="imagenet_c",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCGaussianNoise(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with Gaussian noise."""

    resource: Tuple[str, ...] = variable(
        (
            "gaussian_noise",
            "noise.tar",
            "https://zenodo.org/record/2235448/files/noise.tar?download=1",
            "e80562d7f6c3f8834afb1ecf27252745",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Shot Noise)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCShotNoise(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with shot noise."""

    resource: Tuple[str, ...] = variable(
        (
            "shot_noise",
            "noise.tar",
            "https://zenodo.org/record/2235448/files/noise.tar?download=1",
            "e80562d7f6c3f8834afb1ecf27252745",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Impulse Noise)",
    relative_data_folder="imagenet_c",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCImpulseNoise(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with impulse noise."""

    resource: Tuple[str, ...] = variable(
        (
            "impulse_noise",
            "noise.tar",
            "https://zenodo.org/record/2235448/files/noise.tar?download=1",
            "e80562d7f6c3f8834afb1ecf27252745",
        )
    )


# blur corruptions
@sh_benchmark.register_task(
    name="ImageNet-C (Defocus Blur)",
    relative_data_folder="imagenet_c",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCDefocusBlur(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by defocus blur."""

    resource: Tuple[str, ...] = variable(
        (
            "defocus_blur",
            "blur.tar",
            "https://zenodo.org/record/2235448/files/blur.tar?download=1",
            "2d8e81fdd8e07fef67b9334fa635e45c",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Glass Blur)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCGlassBlur(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by glass blur."""

    resource: Tuple[str, ...] = variable(
        (
            "glass_blur",
            "blur.tar",
            "https://zenodo.org/record/2235448/files/blur.tar?download=1",
            "2d8e81fdd8e07fef67b9334fa635e45c",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Motion Blur)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCMotionBlur(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by motion blur."""

    resource: Tuple[str, ...] = variable(
        (
            "motion_blur",
            "blur.tar",
            "https://zenodo.org/record/2235448/files/blur.tar?download=1",
            "2d8e81fdd8e07fef67b9334fa635e45c",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Zoom Blur)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCZoomBlur(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by zoom blur."""

    resource: Tuple[str, ...] = variable(
        (
            "zoom_blur",
            "blur.tar",
            "https://zenodo.org/record/2235448/files/blur.tar?download=1",
            "2d8e81fdd8e07fef67b9334fa635e45c",
        )
    )


# weather corruptions
@sh_benchmark.register_task(
    name="ImageNet-C (Brightness)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCBrightness(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by brightness changes."""

    resource: Tuple[str, ...] = variable(
        (
            "brightness",
            "weather.tar",
            "https://zenodo.org/record/2235448/files/weather.tar?download=1",
            "33ffea4db4d93fe4a428c40a6ce0c25d",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Frost)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCFrost(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with frost."""

    resource: Tuple[str, ...] = variable(
        (
            "frost",
            "weather.tar",
            "https://zenodo.org/record/2235448/files/weather.tar?download=1",
            "33ffea4db4d93fe4a428c40a6ce0c25d",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Snow)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCSnow(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with snow."""

    resource: Tuple[str, ...] = variable(
        (
            "snow",
            "weather.tar",
            "https://zenodo.org/record/2235448/files/weather.tar?download=1",
            "33ffea4db4d93fe4a428c40a6ce0c25d",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Fog)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCFog(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with fog."""

    resource: Tuple[str, ...] = variable(
        (
            "fog",
            "weather.tar",
            "https://zenodo.org/record/2235448/files/weather.tar?download=1",
            "33ffea4db4d93fe4a428c40a6ce0c25d",
        )
    )


# digital corruptions
@sh_benchmark.register_task(
    name="ImageNet-C (Contrast)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCContrast(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by contrast changes."""

    resource: Tuple[str, ...] = variable(
        (
            "contrast",
            "digital.tar",
            "https://zenodo.org/record/2235448/files/digital.tar?download=1",
            "89157860d7b10d5797849337ca2e5c03",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Elastic Transform)",
    relative_data_folder="imagenet_c",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCElasticTransform(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by an elastic transformation."""

    resource: Tuple[str, ...] = variable(
        (
            "elastic_transform",
            "digital.tar",
            "https://zenodo.org/record/2235448/files/digital.tar?download=1",
            "89157860d7b10d5797849337ca2e5c03",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (Pixelate)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCPixelate(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by pixelation."""

    resource: Tuple[str, ...] = variable(
        (
            "pixelate",
            "digital.tar",
            "https://zenodo.org/record/2235448/files/digital.tar?download=1",
            "89157860d7b10d5797849337ca2e5c03",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C (JPEG)", relative_data_folder="imagenet_c", standalone=False
)
@dataclasses.dataclass
class ImageNetCJPEG(ImageNetSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by JPEG compression."""

    resource: Tuple[str, ...] = variable(
        (
            "jpeg_compression",
            "digital.tar",
            "https://zenodo.org/record/2235448/files/digital.tar?download=1",
            "89157860d7b10d5797849337ca2e5c03",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-C", relative_data_folder="imagenet_c", standalone=True
)
@dataclasses.dataclass
class ImageNetCSeparateCorruptions(Task):
    """Classification task on the ImageNet-C dataset where models might use all images
    from one corruption type to adapt in advance.
    """

    max_batch_size: Optional[int] = parameter(
        default=typing.cast(Optional[int], None),
        options=(1, 16, None),
        description="maximum size of batches fed to the model during evaluation",
    )

    # TODO: Add all corruption types here
    corruption_task_cls: Tuple[Type[ImageNetSingleCorruptionTypeBase], ...] = variable(
        (ImageNetCZoomBlur,)
    )

    flavored_corruption_tasks: List[ImageNetSingleCorruptionTypeBase] = variable([])

    def setup(self):
        """Load and prepare data for all child tasks."""

        for corruption_task_cls in self.corruption_task_cls:
            self.flavored_corruption_tasks += list(
                corruption_task_cls.iterate_flavours(data_root=self.data_root)
            )
        for flavored_corruption_task in self.flavored_corruption_tasks:
            flavored_corruption_task.setup()

    def _prepare_dataloader(self) -> Optional[DataLoader]:
        return None

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        results = {}
        accuracies, mces = [], []
        for flavored_corruption_task in self.flavored_corruption_tasks:
            dl = flavored_corruption_task._prepare_dataloader()
            if dl is not None:
                model.prepare(dl)
            corruption_result = flavored_corruption_task.evaluate(model)
            if corruption_result is None:
                # model is not compatible with a subtask and the result should be ignored
                continue

            corruption_name = getattr(
                flavored_corruption_task,
                shifthappens.task_data.task_metadata._TASK_METADATA_FIELD,
            )  # type: ignore
            results[f"accuracy_{corruption_name}"] = corruption_result["accuracy"]
            results[f"mCE_{corruption_name}"] = corruption_result["mce"]

            accuracies.append(corruption_result.accuracy)
            mces.append(corruption_result.mce)

        return TaskResult(
            **results,
            accuracy=np.mean(accuracies).item(),
            mce=np.mean(mces).item(),
            summary_metrics={Metric.Robustness: "accuracy"},
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    sh_benchmark.evaluate_model(ResNet18(device="cpu", max_batch_size=128), "test_data")
