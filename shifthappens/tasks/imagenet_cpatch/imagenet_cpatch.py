"""A task for evaluating the classification accuracy on ImageNet_CPatch.
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
class ImageNetCPatchSingleCorruptionTypeBase(Task):
    """Evaluate the classification accuracy on a single corruption type
     of the ImageNet_CPatch dataset [1,2]. Each corruption type has 5 different
     severity levels. The raw images (before corruptions) in this dataset
     come from the validation set of ImageNet.

    [1] Evaluating Model Robustness to Patch Perturbations.
    Jindong Gu, Volker Tresp, Yao Qin. 2022
    [2] Are Vision Transformers Robust to Patch Perturbations?
    Jindong Gu, Volker Tresp, Yao Qin. ECCV, 2022
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

        accuracy = np.equal(
            all_predicted_labels,
            np.array(self.ch_dataset.targets)[: len(all_predicted_labels)],
        ).mean()

        return TaskResult(
            accuracy=accuracy,
            mce=1.0 - accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


# noise corruptions
@sh_benchmark.register_task(
    name="ImageNet_CPatch (Gaussian Noise)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchGaussianNoise(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise Gaussian noise."""

    resource: Tuple[str, ...] = variable(
        (
            "gaussian_noise",
            "noise_cpatch.tar",
            "https://zenodo.org/record/7378906/files/noise_cpatch.tar?download=1",
            "md5:8ccb83d0cf614d46903499517a08e2c9",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Shot Noise)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchShotNoise(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise shot noise."""

    resource: Tuple[str, ...] = variable(
        (
            "shot_noise",
            "noise_cpatch.tar",
            "https://zenodo.org/record/7378906/files/noise_cpatch.tar?download=1",
            "md5:8ccb83d0cf614d46903499517a08e2c9",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Impulse Noise)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchImpulseNoise(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise impulse noise."""

    resource: Tuple[str, ...] = variable(
        (
            "impulse_noise",
            "noise_cpatch.tar",
            "https://zenodo.org/record/7378906/files/noise_cpatch.tar?download=1",
            "md5:8ccb83d0cf614d46903499517a08e2c9",
        )
    )


# blur corruptions
@sh_benchmark.register_task(
    name="ImageNet_CPatch (Defocus Blur)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchDefocusBlur(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise defocus blur."""

    resource: Tuple[str, ...] = variable(
        (
            "defocus_blur",
            "blur_cpatch.tar",
            "https://zenodo.org/record/7378906/files/blur_cpatch.tar?download=1",
            "md5:8cf5c9bf0958df6676d32af762bcf35b",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Glass Blur)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchGlassBlur(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise glass blur."""

    resource: Tuple[str, ...] = variable(
        (
            "glass_blur",
            "blur_cpatch.tar",
            "https://zenodo.org/record/7378906/files/blur_cpatch.tar?download=1",
            "md5:8cf5c9bf0958df6676d32af762bcf35b",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Motion Blur)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchMotionBlur(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise motion blur."""

    resource: Tuple[str, ...] = variable(
        (
            "motion_blur",
            "blur_cpatch.tar",
            "https://zenodo.org/record/7378906/files/blur_cpatch.tar?download=1",
            "md5:8cf5c9bf0958df6676d32af762bcf35b",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Zoom Blur)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchZoomBlur(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise zoom blur."""

    resource: Tuple[str, ...] = variable(
        (
            "zoom_blur",
            "blur_cpatch.tar",
            "https://zenodo.org/record/7378906/files/blur_cpatch.tar?download=1",
            "md5:8cf5c9bf0958df6676d32af762bcf35b",
        )
    )


# weather corruptions
@sh_benchmark.register_task(
    name="ImageNet_CPatch (Brightness)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchBrightness(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise brightness changes."""

    resource: Tuple[str, ...] = variable(
        (
            "brightness",
            "digital_cpatch.tar",
            "https://zenodo.org/record/7378906/files/weather_cpatch.tar?download=1",
            "md5:15262cd43f4d80bba111d9bba8f1451a",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Frost)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchFrost(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise frost."""

    resource: Tuple[str, ...] = variable(
        (
            "frost",
            "weather_cpatch.tar",
            "https://zenodo.org/record/7378906/files/weather_cpatch.tar?download=1",
            "md5:bb65bcad0f9b60bcca710a1b263c8ad4",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Snow)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchSnow(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise snow."""

    resource: Tuple[str, ...] = variable(
        (
            "snow",
            "weather_cpatch.tar",
            "https://zenodo.org/record/7378906/files/weather_cpatch.tar?download=1",
            "md5:bb65bcad0f9b60bcca710a1b263c8ad4",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Fog)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchFog(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed with patch-wise fog."""

    resource: Tuple[str, ...] = variable(
        (
            "fog",
            "weather_cpatch.tar",
            "https://zenodo.org/record/7378906/files/weather_cpatch.tar?download=1",
            "md5:bb65bcad0f9b60bcca710a1b263c8ad4",
        )
    )


# digital corruptions
@sh_benchmark.register_task(
    name="ImageNet_CPatch (Contrast)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchContrast(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise contrast changes."""

    resource: Tuple[str, ...] = variable(
        (
            "contrast",
            "digital_cpatch.tar",
            "https://zenodo.org/record/7378906/files/digital_cpatch.tar?download=1",
            "md5:15262cd43f4d80bba111d9bba8f1451a",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Elastic Transform)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchElasticTransform(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by a patch-wise elastic transformation."""

    resource: Tuple[str, ...] = variable(
        (
            "elastic_transform",
            "digital_cpatch.tar",
            "https://zenodo.org/record/7378906/files/digital_cpatch.tar?download=1",
            "md5:15262cd43f4d80bba111d9bba8f1451a",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (Pixelate)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchPixelate(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise pixelation."""

    resource: Tuple[str, ...] = variable(
        (
            "pixelate",
            "digital_cpatch.tar",
            "https://zenodo.org/record/7378906/files/digital_cpatch.tar?download=1",
            "md5:15262cd43f4d80bba111d9bba8f1451a",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet_CPatch (JPEG)",
    relative_data_folder="imagenet_cpatch",
    standalone=False,
)
@dataclasses.dataclass
class ImageNetCPatchJPEG(ImageNetCPatchSingleCorruptionTypeBase):
    """Evaluate classification accuracy on validation images of ImageNet
    perturbed by patch-wise JPEG compression."""

    resource: Tuple[str, ...] = variable(
        (
            "jpeg_compression",
            "digital_cpatch.tar",
            "https://zenodo.org/record/7378906/files/digital_cpatch.tar?download=1",
            "md5:15262cd43f4d80bba111d9bba8f1451a",
        )
    )


@sh_benchmark.register_task(
    name="ImageNetCPatch", relative_data_folder="imagenet_cpatch", standalone=True
)
@dataclasses.dataclass
class ImageNetCPatchSeparateCorruptions(Task):
    """Classification task on the ImageNet_CPatch dataset where models might use all images
    from one corruption type to adapt in advance.
    """

    max_batch_size: Optional[int] = parameter(
        default=typing.cast(Optional[int], None),
        options=(1, 16, None),
        description="maximum size of batches fed to the model during evaluation",
    )

    corruption_task_cls: Tuple[Type[ImageNetCPatchSingleCorruptionTypeBase], ...] = variable(
        (ImageNetCPatchGaussianNoise, ImageNetCPatchShotNoise, ImageNetCPatchImpulseNoise, ImageNetCPatchDefocusBlur,
        ImageNetCPatchGlassBlur, ImageNetCPatchMotionBlur, ImageNetCPatchZoomBlur, ImageNetCPatchBrightness,
        ImageNetCPatchFrost, ImageNetCPatchSnow, ImageNetCPatchFog, ImageNetCPatchContrast, ImageNetCPatchPixelate,
        ImageNetCPatchJPEG)
    )

    flavored_corruption_tasks: List[ImageNetCPatchSingleCorruptionTypeBase] = variable([])

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
            accuracy=np.mean(np.array(accuracies)),
            mce=np.mean(np.array(mces)),
            summary_metrics={Metric.Robustness: "accuracy"},
        )
