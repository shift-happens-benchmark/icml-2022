"""Classifiers Should Do Well Even on Their Worst Classes"""
import collections
import dataclasses
import os
import pathlib
import time
import urllib
from typing import List
from typing import Union

import numpy as np
import requests
import worst_case_utils
from numpy.core.multiarray import ndarray

import shifthappens.config
from shifthappens import benchmark as sh_benchmark
from shifthappens.data import imagenet as sh_imagenet
from shifthappens.models import base as sh_models
from shifthappens.tasks.base import parameter
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

    probs = None
    labels_type: str = parameter(
        default="val",
        options=("val", "val_clean"),
        description="set the label type either to 50000 or 46044 for the "
        "cleaned labels from [3]",
    )
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

        if self.labels_type == "val_clean":
            cleaned_labels = new_labels != -1
            self.new_labels = new_labels[cleaned_labels]
        elif self.labels_type == "val":
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

    def _evaluate(self, model: sh_models.Model) -> TaskResult:
        """The final method that uses all of the above to compute the metrics introduced in [1]"""
        verbose = shifthappens.config.verbose

        if verbose:
            print(
                f"new labels of type {self.labels_type} are",
                self.new_labels,
                len(self.new_labels),
            )

        self.probs = model.imagenet_validation_result.confidences[
            self.new_labels_mask, :
        ]
        preds = self.get_predictions()
        classwise_accuracies_dict = worst_case_utils.classwise_accuracies(
            preds, self.new_labels
        )

        metrics = {
            "A": worst_case_utils.standard_accuracy(preds, self.new_labels),
            "WCA": worst_case_utils.worst_class_accuracy(classwise_accuracies_dict),
            "WCP": worst_case_utils.worst_class_precision(preds, self.new_labels),
            "WSupCA": worst_case_utils.worst_intra_superclass_accuracy(
                self.probs, self.new_labels, self.superclasses
            ),
            "WSupCR": worst_case_utils.worst_superclass_recall(
                preds, self.new_labels, self.superclasses
            ),
            "W10CR": worst_case_utils.worst_heuristic_n_classes_recall(
                preds, self.new_labels, 10
            ),
            "W100CR": worst_case_utils.worst_heuristic_n_classes_recall(
                preds, self.new_labels, 100
            ),
            "W2CA": worst_case_utils.worst_balanced_two_class_binary_accuracy(
                self.probs, self.new_labels
            ),
            "WCAat5": worst_case_utils.worst_class_topk_accuracy(
                preds, self.new_labels, 5
            ),
            "W10CRat5": worst_case_utils.worst_heuristic_n_classes_topk_recall(
                preds, self.new_labels, 10, 5
            ),
            "W100CRat5": worst_case_utils.worst_heuristic_n_classes_topk_recall(
                preds, self.new_labels, 100, 5
            ),
        }

        if verbose:
            print("metrics are", metrics)
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
            **metrics,
        )
