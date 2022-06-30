"""Shift Happens task: ImageNet-M"""

import dataclasses
import json
import os

import numpy as np
import torchvision.datasets.utils as tv_utils
from torchvision import datasets as tv_datasets

from shifthappens import benchmark as sh_benchmark
from shifthappens.models import base as sh_models
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


HUMAN_ACCURACY_JSON = "https://storage.googleapis.com/brain-car-datasets/imagenet-mistakes/human_accuracy_v3.0.0.json"
CLASS_INFO_JSON = "https://raw.githubusercontent.com/modestyachts/ImageNetV2/master/data/metadata/class_info.json"
IMAGENET_VALIDATION_LABELS = "https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image_classification/imagenet2012_validation_labels.txt"

# If true, loads Greedy Soups model logits (pre-computed).
# Eval should return 19 correct ImageNet-M examples.
LOAD_GREEDYSOUPS_LOGITS = False
GREEDYSOUPS_JSON = "https://storage.googleapis.com/brain-car-datasets/imagenet-mistakes/logits/greedysoups.npz"


def load_npy_as_dict(path):
    """Loads a numpy array of logits as a dictionary.

    The returned dictionary maps ImageNet validation filenames
    to a numpy array of logit values.
    """
    logits_array = np.load(open(path, "rb"))
    ret = {}
    for i in range(len(logits_array)):
        filename = f"ILSVRC2012_val_{i+1:08}.JPEG".encode()
        ret[filename] = logits_array[i]
    return ret


@sh_benchmark.register_task(
    name="ImageNet-M", relative_data_folder="imagenet_m", standalone=True
)
@dataclasses.dataclass
class ImageNetM(Task):
    def setup(self):
        dataset_folder = os.path.join(self.data_root, "imagenet-m")
        print("Downloading to ", dataset_folder)
        tv_utils.download_url(
            HUMAN_ACCURACY_JSON,
            dataset_folder,
            "human_accuracy.json",
            "81d98aa82e570a3b2db98af5ccf87145",
        )
        tv_utils.download_url(
            CLASS_INFO_JSON,
            dataset_folder,
            "class_info.json",
            "50608bba554770c263cc8e23a2903c1c",
        )
        tv_utils.download_url(
            IMAGENET_VALIDATION_LABELS,
            dataset_folder,
            "val_labels.txt",
            "eaa6788f6a51fca5fb184f235d0c4e62",
        )
        if LOAD_GREEDYSOUPS_LOGITS:
            tv_utils.download_url(
                GREEDYSOUPS_JSON,
                dataset_folder,
                "greedysoups.npz",
                "736b127e29091f01737c3e26a212ee0d",
            )

        human_accuracy = json.load(
            open(os.path.join(dataset_folder, "human_accuracy.json"))
        )

        imagenet_class_info = json.load(
            open(os.path.join(dataset_folder, "class_info.json"))
        )

        # Create mapping from wnid to cid.
        self._wnid_to_cid = {}
        for entry in imagenet_class_info:
            self._wnid_to_cid[entry["wnid"]] = entry["cid"]

        # Load validation targets.
        self._labels_list = (
            open(os.path.join(dataset_folder, "val_labels.txt"))
            .read()
            .strip()
            .splitlines()
        )

        # This contains the list of the 68 files comprising the ImageNet-M task.
        self._imagenet_m_2022_files = set(human_accuracy["imagenet_m"])

        # Load mapping of filename to the set of correct multilabels.
        self._imagenet_m_2022 = {}
        for image_name, data in human_accuracy["initial_annots"].items():
            if image_name not in self._imagenet_m_2022_files:
                continue
            correct_multi_labels = data.get("correct", [])
            unclear_multi_labels = data.get("unclear", [])
            correct_cids = [self._wnid_to_cid[x] for x in correct_multi_labels]
            unclear_cids = [self._wnid_to_cid[x] for x in unclear_multi_labels]
            entry = {
                "correct_multi_labels": correct_cids,
                "unclear_multi_labels": unclear_cids,
            }
            self._imagenet_m_2022[image_name] = entry

        if LOAD_GREEDYSOUPS_LOGITS:
            self._greedy_soups = load_npy_as_dict(
                os.path.join(dataset_folder, "greedysoups.npz")
            )
        else:
            # Create mapping from torch DataLoader order to filename
            self._filename_to_pred_index = {}
            iif = tv_datasets.ImageFolder(root=imagenet.ImageNetValidationData)
            for i, (filename, _) in enumerate(iif.samples):
                im_filename = filename.split("/")[-1]
                self._filename_to_pred_index[im_filename] = i

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        # Get imagenet validation results for the model.
        imagenet_val_data = model.imagenet_validation_result

        # For every image in imagenet_m, check whether prediction
        # is in correct set for ImageNet-M.
        correct = 0
        for image_name, data in self._imagenet_m_2022.items():
            image_id = int(image_name.split("_")[-1][:-5])
            if LOAD_GREEDYSOUPS_LOGITS:
                logits = self._greedy_soups[image_name.encode()]
                predicted_class = np.argmax(logits)
            else:
                pred_index = self._filename_to_pred_index[image_name]
                predicted_class = imagenet_val_data.class_labels[pred_index]

            if (
                predicted_class in data["correct_multi_labels"]
                or predicted_class in data["unclear_multi_labels"]
            ):
                correct += 1

        print("Total ImageNet-M correct:", correct)

        return TaskResult(
            accuracy=correct / len(self._imagenet_m_2022),
            summary_metrics={Metric.Robustness: "accuracy"},
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18

    result = sh_benchmark.evaluate_model(
        ResNet18(device="cpu", max_batch_size=128), "data"
    )
    print("ImageNet-M accuracy: ", list(result.values())[0]._metrics["accuracy"])
