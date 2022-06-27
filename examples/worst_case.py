"""
This task evaluates a set of metrics, mostly related to worst-class performance, as described in
(J. Bitterwolf et al., "Classifiers Should Do Well Even on Their Worst Classes", https://openreview.net/forum?id=QxIXCVYJ2WP).
It is motivated by
(R. Balestriero et al., "The Effects of Regularization and Data Augmentation are Class Dependent", https://arxiv.org/abs/2204.03632)
 where the authors note that using only accuracy as a metric is not enough to evaluate
 the performance of the classifier, as it must not be the same on all classes/groups.
"""
import argparse
import collections
import itertools
import time
import urllib
import  requests
import dataclasses
import os
import pathlib

import numpy as np

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
    resources = (
        ["worstcase",
        "restricted_superclass.csv",
        "https://anonymous.4open.science/r/worst_classes-B94C/restricted_superclass.csv",
        None],

        ["worstcase",
         "new_labels.csv",
         "https://anonymous.4open.science/r/worst_classes-B94C/new_labels.csv",
         None]
    )

    new_labels = None
    new_labels_mask = None
    superclasses = None

    verbose = True
    labels_type = 'val'
    n_retries = 5
    max_batch_size: int = 256

    def download(self, url, data_folder, filename, md5):

        for _ in range(self.n_retries):
            try:
                r = requests.get(url)
                pathlib.Path(data_folder).mkdir(parents=True, exist_ok=True)
                open(os.path.join(data_folder, filename), 'wb').write(r.content)
                break
            except urllib.error.URLError:
                print(f"Download of {url} failed; wait 5s and then try again.")
                time.sleep(5)
    def setup(self):

        # Download resources
        for resource in self.resources:
            folder_name, file_name, url, md5 = resource
            dataset_folder = os.path.join(self.data_root, folder_name)
            if not os.path.isfile(os.path.join(dataset_folder, file_name)):
                self.download(url, dataset_folder, file_name, md5)
            print(f'File {file_name} is in {dataset_folder}.')
        # Set the cleaned labels to a property
        new_labels = np.array([int(line) for line in open(os.path.join(dataset_folder, 'new_labels.csv'))])

        if self.labels_type == 'val_clean':
            gooduns = new_labels != -1
            self.new_labels = new_labels[gooduns]
        elif self.labels_type == 'val':
            gooduns = np.full(new_labels.shape, True)
            self.new_labels = np.array(sh_imagenet.load_imagenet_targets())

        self.new_labels_mask = gooduns

        # Set the superclasses to a property
        superclass_list = np.array([int(line) for line in open(os.path.join(dataset_folder, 'restricted_superclass.csv'))])
        self.superclasses = [tuple(np.where(superclass_list == i)[0]) for i in range(0, 9)]

    def get_predictions(self) -> np.ndarray:
        preds = {
                 'predicted_classes': self.probs.argmax(axis=1),
                 'class_probabilities': self.probs,
                 'confidences_classifier': self.probs.max(axis=1),
                 }
        preds['number_of_class_predictions'] = collections.Counter(preds['predicted_classes'])
        return preds

    def standard_accuracy(self):
        preds = self.get_predictions()
        accuracy = (preds['predicted_classes'] == self.new_labels).mean()
        return accuracy

    def classwise_accuracies(self):
        preds = self.get_predictions()
        clw_acc = {}
        for i in set(self.new_labels):
            clw_acc[i] = np.equal(preds['predicted_classes'][np.where(self.new_labels == i)], i).mean()
        return clw_acc

    def classwise_sample_numbers(self):
        clw_sn = {}
        for i in set(self.new_labels):
            clw_sn[i] = np.sum(self.new_labels == i)
        return clw_sn

    def classwise_topk_accuracies(self, k):
        preds = self.get_predictions()
        clw_topk_acc = {}
        for i in set(self.new_labels):
            clw_topk_acc[i] = np.equal(i, np.argsort(preds['class_probabilities'][np.where(self.new_labels == i)], axis=1, kind='mergesort')[:,
                                          -k:]).sum(axis=-1).mean()
        return clw_topk_acc

    def standard_balanced_topk_accuracy(self, k):
        clw_topk_acc = self.classwise_topk_accuracies(k)
        return np.array(list(clw_topk_acc.values())).mean()

    def worst_class_accuracy(self):
        cwa = self.classwise_accuracies()
        worst_item = min(cwa.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_class_topk_accuracy(self, k):
        clw_topk_acc = self.classwise_topk_accuracies(k)
        worst_item = min(clw_topk_acc.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_balanced_n_classes_accuracy(self, n):
        cwa = self.classwise_accuracies()
        sorted_cwa = sorted(cwa.items(), key=lambda item: item[1])
        n_worst = sorted_cwa[:n]
        return np.array([x[1] for x in n_worst]).mean()

    def worst_heuristic_n_classes_recall(self, n):
        cwa = self.classwise_accuracies()
        clw_sn = self.classwise_sample_numbers()
        sorted_cwa = sorted(cwa.items(), key=lambda item: item[1])
        n_worst = sorted_cwa[:n]
        nwc = np.array([v * clw_sn[c] for c, v in n_worst]).sum() / np.array([clw_sn[c] for c, v in n_worst]).sum()
        return nwc

    def worst_balanced_n_classes_topk_accuracy(self, n, k):
        clw_topk_acc = self.classwise_topk_accuracies(k)
        sorted_clw_topk_acc = sorted(clw_topk_acc.items(), key=lambda item: item[1])
        n_worst = sorted_clw_topk_acc[:n]
        return np.array([x[1] for x in n_worst]).mean()

    def worst_heuristic_n_classes_topk_recall(self, n, k):
        clw_topk_acc = self.classwise_topk_accuracies(k)
        clw_sn = self.classwise_sample_numbers()
        sorted_clw_topk_acc = sorted(clw_topk_acc.items(), key=lambda item: item[1])
        n_worst = sorted_clw_topk_acc[:n]
        nwc = np.array([v * clw_sn[c] for c, v in n_worst]).sum() / np.array([clw_sn[c] for c, v in n_worst]).sum()
        return nwc

    def worst_balanced_two_class_binary_accuracy(self):
        classes = list(set(self.new_labels))
        binary_accuracies = {}
        for i, j in itertools.combinations(classes, 2):
            i_labelled = self.probs[np.where(self.new_labels == i)]
            j_labelled = self.probs[np.where(self.new_labels == j)]
            i_correct = np.greater(i_labelled[:, i], i_labelled[:, j]).mean()
            j_correct = np.greater(j_labelled[:, j], j_labelled[:, i]).mean()
            binary_accuracies[(i, j)] = (i_correct + j_correct) / 2
        sorted_binary_accuracies = sorted(binary_accuracies.items(), key=lambda item: item[1])
        worst_item = sorted_binary_accuracies[0]
        return worst_item[1]

    def worst_balanced_superclass_recall(self):
        cwa = self.classwise_accuracies()
        scwa = {i: np.array([cwa[c] for c in s]).mean() for i, s in enumerate(self.superclasses)}
        worst_item = min(scwa.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_superclass_recall(self):
        cwa = self.classwise_accuracies()
        clw_sn = self.classwise_sample_numbers()
        scwa = {i: np.array([cwa[c] * clw_sn[c] for c in s]).sum() / np.array([clw_sn[c] for c in s]).sum() for i, s in
                enumerate(self.superclasses)}
        worst_item = min(scwa.items(), key=lambda x: x[1])
        return worst_item[1]

    def intra_superclass_accuracies(self):
        isa = {}
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
            internal_preds = s_targets(self.get_predictions()['predicted_classes'])
            isa[i] = (internal_preds == internal_targets).mean()

        self.probs = original_probs
        self.new_labels = original_targets

        return isa

    def worst_intra_superclass_accuracy(self):
        isa = self.intra_superclass_accuracies()
        worst_item = min(isa.items(), key=lambda x: x[1])
        return worst_item[1]

    def worst_class_precision(self):
        preds = self.get_predictions()
        classes = list(set(self.new_labels))
        sc = {}
        for c in classes:
            erroneous_c = (preds['predicted_classes'] == c) * (self.new_labels != c)
            correct_c = (preds['predicted_classes'] == c) * (self.new_labels == c)
            predicted_c = (preds['predicted_classes'] == c)
            if predicted_c.sum():
                sc[c] = correct_c.sum() / predicted_c.sum()  # 1-erroneous_c.sum()/predicted_c.sum()
            else:
                sc[c] = 1
        sorted_sc = sorted(sc.items(), key=lambda item: item[1])
        worst_item = sorted_sc[0]
        return worst_item[1]

    def class_confusion(self):
        preds = self.get_predictions()
        classes = list(set(self.new_labels))
        confusion = np.zeros((len(classes), len(classes)))
        for i, c in enumerate(self.new_labels):
            confusion[c, preds['predicted_classes'][i]] += 1
        return confusion


    def _evaluate(self, model: sh_models.Model, verbose=False) -> TaskResult:

        verbose = self.verbose
        model.verbose = verbose

        if verbose:
            print(f'new labels of type {self.labels_type} are', self.new_labels, len(self.new_labels))

        self.probs = model.imagenet_validation_result.confidences[self.new_labels_mask, :]

        metrics = {
            'A': self.standard_accuracy,
            'WCA': self.worst_class_accuracy,
            'WCP': self.worst_class_precision,
            'WSupCA': self.worst_intra_superclass_accuracy,
            'WSupCR': self.worst_superclass_recall,
            'W10CR': lambda : self.worst_heuristic_n_classes_recall(10),
            'W100CR': lambda : self.worst_heuristic_n_classes_recall(100),
            'W2CA': self.worst_balanced_two_class_binary_accuracy,
            'WCAat5': lambda : self.worst_class_topk_accuracy(5),
            'W10CRat5': lambda : self.worst_heuristic_n_classes_topk_recall(10, 5),
            'W100CRat5': lambda : self.worst_heuristic_n_classes_topk_recall(100, 5),
        }

        metrics_eval = {}
        for metric_name, metric in metrics.items():
            if verbose:
                print(f'Evaluating {metric_name}')
            metrics_eval[metric_name] = metric()
        if verbose:
            print('metrics are', metrics_eval)
        return TaskResult(
            summary_metrics={Metric.Fairness: ("A", "WCA", "WCP", "WSupCA", "WSupCR",
                                               "W10CR", "W100CR", "W2CA", "WCAat5",
                                               "W10CRat5", "W100CRat5")},
            **metrics_eval

        )


if __name__ == "__main__":
    from shifthappens.models.torchvision import ResNet18
    import shifthappens

    parser = argparse.ArgumentParser()

    # Set the label type either to val (50000 labels) or
    # val_clean (46044 labels) for the cleaned labels from
    # (C. Northcutt et al., "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", https://arxiv.org/abs/2103.14749, https://github.com/cleanlab/label-errors)
    parser.add_argument(
        "--labels_type", type=str, help="The label type", default='val'
    )
    parser.add_argument(
        "--imagenet_val_folder", type=str, help="The folder for the imagenet val set", required=True
    )
    parser.add_argument(
        "--verbose",
        help="Turn verbose mode on when set",
        action="store_true",
    )

    args = parser.parse_args()

    shifthappens.data.imagenet.ImageNetValidationData=args.imagenet_val_folder


    tuple(sh_benchmark.__registered_tasks)[0].cls.verbose = args.verbose

    tuple(sh_benchmark.__registered_tasks)[0].cls.labels_type = args.labels_type
    sh_benchmark.evaluate_model(
        ResNet18(device="cuda:2", max_batch_size=500),
        "data",
        verbose=args.verbose
    )
