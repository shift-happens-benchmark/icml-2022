"""Classifiers Should Do Well Even on Their Worst Classes"""
import argparse
import collections
import itertools
import time
import urllib
from typing import List, Union

import requests
import dataclasses
import os
import pathlib
import torch
import numpy as np
import torch.nn as nn
import shifthappens.config
from numpy.core.multiarray import ndarray

from shifthappens.data import imagenet as sh_imagenet
from shifthappens import benchmark as sh_benchmark
from shifthappens.models import base as sh_models
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


@sh_benchmark.register_task(
    name="Worst_case", relative_data_folder="worst_case", standalone=True
)
@dataclasses.dataclass
class WorstCase(Task):
    """This task evaluates a set of metrics, mostly related to worst-class performance, as described in [1].
    It is motivated by [2], where the authors note that using only accuracy as a metric is not enough to evaluate
    the performance of the classifier, as it must not be the same on all classes/groups."""

    resources = (
        [
            "worstcase",
            "restricted_superclass.csv",
            "https://anonymous.4open.science/r/worst_classes-B94C/restricted_superclass.csv",
            None,
        ],
        [
            "worstcase",
            "new_labels.csv",
            "https://anonymous.4open.science/r/worst_classes-B94C/new_labels.csv",
            None,
        ],
    )

    new_labels = None
    new_labels_mask: Union[ndarray, None, bool] = None
    superclasses: List[tuple] = None

    verbose: bool = True
    probs = None
    # labels_type: str = 'val'
    n_retries: int = 5
    max_batch_size: int = 256

    def download(self, url, data_folder, filename, md5):
        """Method to download the data given its' url, and the desired folder to stor int"""
        for _ in range(self.n_retries):
            try:
                r = requests.get(url)
                pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
                open(os.path.join(data_folder, filename), "wb").write(r.content)
                break
            except urllib.error.URLError:
                print(f"Download of {url} failed; wait 5s and then try again.")
                time.sleep(5)

    def setup(self):
        """Calls the download method to download the cleaned labels from [3], as well as superclasses used in [1]"""
        # Download resources
        for resource in self.resources:
            folder_name, file_name, url, md5 = resource
            dataset_folder = os.path.join(self.data_root, folder_name)
            if not os.path.isfile(os.path.join(dataset_folder, file_name)):
                self.download(url, dataset_folder, file_name, md5)
            print(f"File {file_name} is in {dataset_folder}.")
        # Set the cleaned labels to a property
        new_labels: ndarray = np.array(
            [int(line) for line in open(os.path.join(dataset_folder, "new_labels.csv"))]
        )

        if os.environ["SH_labels_type"] == "val_clean":
            cleaned_labels = new_labels != -1
            self.new_labels = new_labels[cleaned_labels]
        elif os.environ["SH_labels_type"] == "val":
            cleaned_labels = np.full(new_labels.shape, True)
            self.new_labels = np.array(sh_imagenet.load_imagenet_targets())

        self.new_labels_mask = cleaned_labels

        # Set the superclasses to a property
        superclass_list: ndarray = np.array(
            [
                int(line)
                for line in open(
                    os.path.join(dataset_folder, "restricted_superclass.csv")
                )
            ]
        )
        self.superclasses = [
            tuple(np.where(superclass_list == i)[0]) for i in range(0, 9)
        ]

    def get_predictions(self) -> np.ndarray:
        """Saves to a property as a dict the computed predictions and probabilities for the used model"""
        preds = {
            "predicted_classes": self.probs.argmax(axis=1),
            "class_probabilities": self.probs,
            "confidences_classifier": self.probs.max(axis=1),
        }
        preds["number_of_class_predictions"] = collections.Counter(
            preds["predicted_classes"]
        )
        return preds

    def standard_accuracy(self) -> np.float:
        """Computes standard accuracy"""
        preds = self.get_predictions()
        accuracy = (preds["predicted_classes"] == self.new_labels).mean()
        return accuracy

    def classwise_accuracies(self) -> dict:
        """Computes accuracies per each class"""
        preds = self.get_predictions()
        clw_acc = {}
        for i in set(self.new_labels):
            clw_acc[i] = np.equal(
                preds["predicted_classes"][np.where(self.new_labels == i)], i
            ).mean()
        return clw_acc

    def classwise_sample_numbers(self) -> dict:
        """Computes number of samples per class"""
        classwise_sample_number = {}
        for i in set(self.new_labels):
            classwise_sample_number[i] = np.sum(self.new_labels == i)
        return classwise_sample_number

    def classwise_topk_accuracies(self, k) -> dict:
        """Computes topk accuracies per class"""
        preds = self.get_predictions()
        classwise_topk_acc = {}
        for i in set(self.new_labels):
            classwise_topk_acc[i] = (
                np.equal(
                    i,
                    np.argsort(
                        preds["class_probabilities"][np.where(self.new_labels == i)],
                        axis=1,
                        kind="mergesort",
                    )[:, -k:],
                )
                .sum(axis=-1)
                .mean()
            )
        return classwise_topk_acc

    def standard_balanced_topk_accuracy(self, k) -> np.array:
        """Computes the balanced topk accuracy"""
        classwise_topk_acc = self.classwise_topk_accuracies(k)
        return np.array(list(classwise_topk_acc.values())).mean()

    def worst_class_accuracy(self) -> float:
        """Computes the smallest accuracy among classes"""
        classwise_accuracies = self.classwise_accuracies()
        worst_item = min(classwise_accuracies.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_class_topk_accuracy(self, k) -> float:
        """Computes the smallest topk accuracy among classes"""
        classwise_topk_acc = self.classwise_topk_accuracies(k)
        worst_item = min(classwise_topk_acc.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_balanced_n_classes_accuracy(self, n) -> np.array:
        """Computes the ballanced accuracy among the worst n classes, based on their per-class accuracies"""
        classwise_accuracies = self.classwise_accuracies()
        sorted_classwise_accuracies = sorted(
            classwise_accuracies.items(), key=lambda item: item[1]
        )
        n_worst = sorted_classwise_accuracies[:n]
        return np.array([x[1] for x in n_worst]).mean()

    def worst_heuristic_n_classes_recall(self, n) -> np.float:
        """Computes recall for n worst in terms of their per class accuracy"""
        classwise_accuracies = self.classwise_accuracies()
        classwise_accuracies_sample_numbers = self.classwise_sample_numbers()
        sorted_classwise_accuracies = sorted(
            classwise_accuracies.items(), key=lambda item: item[1]
        )
        n_worst = sorted_classwise_accuracies[:n]
        n_worstclass_recall = (
            np.array(
                [v * classwise_accuracies_sample_numbers[c] for c, v in n_worst]
            ).sum()
            / np.array(
                [classwise_accuracies_sample_numbers[c] for c, v in n_worst]
            ).sum()
        )
        return n_worstclass_recall

    def worst_balanced_n_classes_topk_accuracy(self, n, k) -> np.float:
        """Computes the balanced accuracy for the worst n classes in therms of their per class topk accuracy"""
        classwise_topk_accuracies = self.classwise_topk_accuracies(k)
        sorted_clw_topk_acc = sorted(
            classwise_topk_accuracies.items(), key=lambda item: item[1]
        )
        n_worst = sorted_clw_topk_acc[:n]
        return np.array([x[1] for x in n_worst]).mean()

    def worst_heuristic_n_classes_topk_recall(self, n, k) -> np.float:
        """Computes the recall for the worst n classes in therms of their per class topk accuracy"""
        classwise_topk_accuracies = self.classwise_topk_accuracies(k)
        classwise_accuracies_sample_numbers = self.classwise_sample_numbers()
        sorted_clw_topk_acc = sorted(
            classwise_topk_accuracies.items(), key=lambda item: item[1]
        )
        n_worst = sorted_clw_topk_acc[:n]
        n_worstclass_recall = (
            np.array(
                [v * classwise_accuracies_sample_numbers[c] for c, v in n_worst]
            ).sum()
            / np.array(
                [classwise_accuracies_sample_numbers[c] for c, v in n_worst]
            ).sum()
        )
        return n_worstclass_recall

    def worst_balanced_two_class_binary_accuracy(self) -> np.float:
        """Computes the smallest two-class accuracy, when restricting the classifier to any two classes"""
        classes = list(set(self.new_labels))
        binary_accuracies = {}
        for i, j in itertools.combinations(classes, 2):
            i_labelled = self.probs[np.where(self.new_labels == i)]
            j_labelled = self.probs[np.where(self.new_labels == j)]
            i_correct = np.greater(i_labelled[:, i], i_labelled[:, j]).mean()
            j_correct = np.greater(j_labelled[:, j], j_labelled[:, i]).mean()
            binary_accuracies[(i, j)] = (i_correct + j_correct) / 2
        sorted_binary_accuracies = sorted(
            binary_accuracies.items(), key=lambda item: item[1]
        )
        worst_item = sorted_binary_accuracies[0]
        return worst_item[1]

    def worst_balanced_superclass_recall(self) -> np.float:
        """Computes the worst balanced recall among the superclasses"""
        classwise_accuracies = self.classwise_accuracies()
        superclass_classwise_accuracies = {
            i: np.array([classwise_accuracies[c] for c in s]).mean()
            for i, s in enumerate(self.superclasses)
        }
        worst_item = min(superclass_classwise_accuracies.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_superclass_recall(self) -> np.float:
        """Computes the worst not balanced recall among the superclasses"""
        classwise_accuracies = self.classwise_accuracies()
        classwise_sample_number = self.classwise_sample_numbers()
        superclass_classwise_accuracies = {
            i: np.array(
                [classwise_accuracies[c] * classwise_sample_number[c] for c in s]
            ).sum()
            / np.array([classwise_sample_number[c] for c in s]).sum()
            for i, s in enumerate(self.superclasses)
        }
        worst_item = min(superclass_classwise_accuracies.items(), key=lambda x: x[1])
        return worst_item[1]

    def intra_superclass_accuracies(self) -> dict:
        """Computes the accuracy for the images among one superclass, for each superclass"""
        intra_superclass_accuracies = {}
        original_probs = self.probs.copy()
        original_targets = self.new_labels.copy()
        for i, s in enumerate(self.superclasses):
            self.probs = original_probs.copy()
            self.new_labels = original_targets.copy()

            internal_samples = np.isin(self.new_labels, s)
            internal_targets = self.new_labels[internal_samples]
            internal_probs = self.probs[internal_samples][:, s]
            s_targets = np.vectorize(lambda x: s[x])
            self.probs = internal_probs
            self.new_labels = internal_targets
            internal_preds = s_targets(self.get_predictions()["predicted_classes"])
            intra_superclass_accuracies[i] = (internal_preds == internal_targets).mean()

        self.probs = original_probs
        self.new_labels = original_targets

        return intra_superclass_accuracies

    def worst_intra_superclass_accuracy(self) -> np.float:
        """Computes the worst superclass accuracy using intra_superclass_accuracies

        Output: the accuracy for the worst super class
        """
        isa = self.intra_superclass_accuracies()
        worst_item = min(isa.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_class_precision(self) -> np.float:
        """Computes the precision for the worst class

        Returns:
           Dict entry with the worst performing class
        """
        preds = self.get_predictions()
        classes = list(set(self.new_labels))
        per_class_precision = {}
        for c in classes:
            erroneous_c = (preds["predicted_classes"] == c) * (self.new_labels != c)
            correct_c = (preds["predicted_classes"] == c) * (self.new_labels == c)
            predicted_c = preds["predicted_classes"] == c
            if predicted_c.sum():
                per_class_precision[c] = (
                    correct_c.sum() / predicted_c.sum()
                )  # 1-erroneous_c.sum()/predicted_c.sum()
            else:
                per_class_precision[c] = 1
        sorted_sc = sorted(per_class_precision.items(), key=lambda item: item[1])
        worst_item = sorted_sc[0]
        return worst_item[1]

    def class_confusion(self) -> np.array:
        """Computes the confision matrix
        Returns:
           confusion: confusion matrx

        """
        preds = self.get_predictions()
        classes = list(set(self.new_labels))
        confusion = np.zeros((len(classes), len(classes)))
        for i, c in enumerate(self.new_labels):
            confusion[c, preds["predicted_classes"][i]] += 1
        return confusion

    def _evaluate(self, model: sh_models.Model, verbose=False) -> TaskResult:
        """The final method that uses all of the above to compute the metrics introduced in [1]"""
        verbose = self.verbose
        model.verbose = verbose

        if verbose:
            print(
                f'new labels of type {os.environ["SH_labels_type"]} are',
                self.new_labels,
                len(self.new_labels),
            )

        self.probs = model.imagenet_validation_result.confidences[
            self.new_labels_mask, :
        ]

        metrics = {
            "A": self.standard_accuracy,
            "WCA": self.worst_class_accuracy,
            "WCP": self.worst_class_precision,
            "WSupCA": self.worst_intra_superclass_accuracy,
            "WSupCR": self.worst_superclass_recall,
            "W10CR": lambda: self.worst_heuristic_n_classes_recall(10),
            "W100CR": lambda: self.worst_heuristic_n_classes_recall(100),
            "W2CA": self.worst_balanced_two_class_binary_accuracy,
            "WCAat5": lambda: self.worst_class_topk_accuracy(5),
            "W10CRat5": lambda: self.worst_heuristic_n_classes_topk_recall(10, 5),
            "W100CRat5": lambda: self.worst_heuristic_n_classes_topk_recall(100, 5),
        }

        metrics_eval = {}
        for metric_name, metric in metrics.items():
            if verbose:
                print(f"Evaluating {metric_name}")
            metrics_eval[metric_name] = metric()
        if verbose:
            print("metrics are", metrics_eval)
        return TaskResult(
            summary_metrics={
                Metric.Fairness: (
                    "A",
                    "WCA",
                    "WCP",
                    "WSupCA",
                    "WSupCR",
                    "W10CR",
                    "W100CR",
                    "W2CA",
                    "WCAat5",
                    "W10CRat5",
                    "W100CRat5",
                )
            },
            **metrics_eval,
        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18, ResNet50, VGG16
    import shifthappens

    available_models_dict = {"resnet18": ResNet18, "resnet50": ResNet50, "vgg16": VGG16}

    parser = argparse.ArgumentParser()

    # Set the label type either to val (50000 labels) or
    # val_clean (46044 labels) for the cleaned labels from
    # [3]
    parser.add_argument("--labels_type", type=str, help="The label type", default="val")
    parser.add_argument(
        "--imagenet_val_folder",
        type=str,
        help="The folder for the imagenet val set",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help=f"The name of the model to test. Should be in {available_models_dict.keys()}",
    )
    parser.add_argument(
        "--gpu",
        "--list",
        nargs="+",
        default=[],
        help="GPU indices, if more than 1 parallel modules will be called",
    )
    parser.add_argument("--bs", type=int, default=500)
    parser.add_argument(
        "--verbose",
        help="Turn verbose mode on when set",
        action="store_true",
    )

    args = parser.parse_args()

    if len(args.gpu) == 0:
        device_ids = None
        device = torch.device("cpu")
        print("Warning! Computing on CPU")
        num_devices = 1
    elif len(args.gpu) == 1:
        device_ids = [int(args.gpu[0])]
        device = torch.device("cuda:" + str(args.gpu[0]))
        num_devices = 1
    else:
        device_ids = [int(i) for i in args.gpu]
        device = torch.device("cuda:" + str(min(device_ids)))
        num_devices = len(device_ids)

    shifthappens.config.imagenet_validation_path = args.imagenet_val_folder

    shifthappens.config.verbose = args.verbose

    os.environ["SH_labels_type"] = args.labels_type

    assert (
        args.model_name.lower() in available_models_dict
    ), f"Selected model_name should be in {available_models_dict.keys()}"

    model = available_models_dict[args.model_name.lower()](
        device=device, max_batch_size=args.bs
    )

    if device_ids is not None and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    sh_benchmark.evaluate_model(model, "data")
