"""Shift Happens task for ImageNet-3DCC
https://3dcommoncorruptions.epfl.ch/

There are 12 corruptions in ImageNet-3DCC. They can be categorized as follows:
Depth of field: Near focus, far focus
Camera motion: XY motion blur, Z motion blur
Lighting: Flash
Video: H256 CRF, H256 ABR, bit error
Weather: Fog 3D
Noise: ISO noise, color quantization, low light
"""

import dataclasses
import os
import typing
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import pickle

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
    resource: Tuple[str, ...] = abstract_variable()

    severity: int = parameter(
        default=1,
        options=(1, 2, 3, 4, 5),
        description="severity of the image corruption",
    )

    max_batch_size: Optional[int] = None

    def setup(self):
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
                tv_transforms.Pad(16),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
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
        
        model.model.eval()

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
        print(self.severity, 1-accuracy.mean())
        return TaskResult(
            accuracy=accuracy,
            mce=1.0 - accuracy,
            summary_metrics={Metric.Robustness: "accuracy"},
        )


@sh_benchmark.register_task(
    name="ImageNet-3DCC (Near Focus)",
    relative_data_folder="imagenet_3dcc",
    standalone=True,
)
@dataclasses.dataclass
class ImageNet3DCCNearFocus(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "near_focus",
            "near_focus.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/near_focus.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Far Focus)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCFarFocus(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "far_focus",
            "far_focus.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/far_focus.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Fog 3D)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCFog3D(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "fog_3d",
            "fog_3d.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/fog_3d.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Flash)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCFlash(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "flash",
            "flash.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/flash.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Color Quant)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCColorQuant(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "color_quant",
            "color_quant.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/flash.tar.gz",
            "tmp",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-3DCC (Low Light)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCLowLight(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "low_light",
            "low_light.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/low_light.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (XY Motion Blur)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCXYMotionBlur(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "xy_motion_blur",
            "xy_motion_blur.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/xy_motion_blur.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Z Motion Blur)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCZMotionBlur(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "z_motion_blur",
            "z_motion_blur.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/z_motion_blur.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (ISO Noise)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCISONoise(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "iso_noise",
            "iso_noise.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/iso_noise.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (Bit Error)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCBitError(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "bit_error",
            "bit_error.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/bit_error.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (H265 ABR)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCH256ABR(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "h265_abr",
            "h265_abr.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/h265_abr.tar.gz",
            "tmp",
        )
    )

@sh_benchmark.register_task(
    name="ImageNet-3DCC (H265 CRF)",
    relative_data_folder="imagenet_3dcc",
    standalone=False,
)
@dataclasses.dataclass
class ImageNet3DCCH256CRF(ImageNetSingleCorruptionTypeBase):
    resource: Tuple[str, ...] = variable(
        (
            "h265_crf",
            "h265_crf.tar.gz",
            "https://datasets.epfl.ch/3dcc/imagenet_3dcc/h265_crf.tar.gz",
            "tmp",
        )
    )


@sh_benchmark.register_task(
    name="ImageNet-3DCC", relative_data_folder="imagenet_3dcc", standalone=False
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

    # Add corruption types to be evaluated here
    corruption_task_cls: Tuple[Type[ImageNetSingleCorruptionTypeBase], ] = variable(
        (ImageNet3DCCFarFocus,)
    )

    flavored_corruption_tasks: List[ImageNetSingleCorruptionTypeBase] = variable([])

    def setup(self):
        
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
    from shifthappens.models.torchvision import resnet18, resnet50

    # sh_benchmark.evaluate_model(resnet18(device="cuda",max_batch_size=2048), "test_data")
    results = sh_benchmark.evaluate_model(resnet50(device="cuda",max_batch_size=2048), "test_data")
    # for k,v in results.items(): print(k,1.-v._metrics["accuracy"].mean())
