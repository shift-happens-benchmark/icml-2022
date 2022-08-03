"""Helper functions for metric calculation for worst case task"""
import itertools

import numpy as np


def standard_accuracy(preds, new_labels) -> np.float64:
    """
    Computes standard accuracy.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
    Returns:
        Standard accuracy value.
    """
    accuracy = (preds["predicted_classes"] == new_labels).mean()
    return accuracy


def classwise_accuracies(preds, new_labels) -> dict:
    """
    Computes accuracies per each class

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
    """
    clw_acc = {}
    for i in set(new_labels):
        clw_acc[i] = np.equal(
            preds["predicted_classes"][np.where(new_labels == i)], i
        ).mean()
    return clw_acc


def classwise_sample_numbers(new_labels) -> dict:
    """
    Computes number of samples per class.

    Args:
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
    """
    classwise_sample_number = {}
    for i in set(new_labels):
        classwise_sample_number[i] = np.sum(new_labels == i)
    return classwise_sample_number


def classwise_topk_accuracies(preds, new_labels, k) -> dict:
    """
    Computes topk accuracies per class

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        k: number of predicted classes at the top of the ranking that used in
            topk accuracy.
    """
    classwise_topk_acc = {}
    for i in set(new_labels):
        classwise_topk_acc[i] = (
            np.equal(
                i,
                np.argsort(
                    preds["class_probabilities"][np.where(new_labels == i)],
                    axis=1,
                    kind="mergesort",
                )[:, -k:],
            )
            .sum(axis=-1)
            .mean()
        )
    return classwise_topk_acc


def worst_balanced_two_class_binary_accuracy(probs, new_labels) -> np.float64:
    """
    Computes the smallest two-class accuracy, when restricting the classifier
    to any two classes.

    Args:
        probs: computed probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
    """
    classes = list(set(new_labels))
    binary_accuracies = {}
    for i, j in itertools.combinations(classes, 2):
        i_labelled = probs[np.where(new_labels == i)]
        j_labelled = probs[np.where(new_labels == j)]
        i_correct = np.greater(i_labelled[:, i], i_labelled[:, j]).mean()
        j_correct = np.greater(j_labelled[:, j], j_labelled[:, i]).mean()
        binary_accuracies[(i, j)] = (i_correct + j_correct) / 2
    sorted_binary_accuracies = sorted(
        binary_accuracies.items(), key=lambda item: item[1]
    )
    worst_item = sorted_binary_accuracies[0]
    return worst_item[1]


def standard_balanced_topk_accuracy(preds, new_labels, k) -> np.array:
    """
    Computes the balanced topk accuracy.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        k: number of predicted classes at the top of the ranking that used in
            topk accuracy.
    """
    classwise_topk_acc = classwise_topk_accuracies(preds, new_labels, k)
    return np.array(list(classwise_topk_acc.values())).mean()


def worst_class_accuracy(classwise_accuracies_dict) -> float:
    """
    Computes the smallest accuracy among classes

    Args:
        classwise_accuracies_dict: computed accuracies per each class.
    """
    worst_item = min(classwise_accuracies_dict.items(), key=lambda x: x[1])
    return worst_item[1]


def worst_class_topk_accuracy(preds, new_labels, k) -> float:
    """
     Computes the smallest topk accuracy among classes.

    Args:
         preds: output of worst_case.WorstCase.get_predictions().
             Predictions and probabilities for the used model.
         new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
         k: number of predicted classes at the top of the ranking that used in
             topk accuracy.
    """
    classwise_topk_acc = classwise_topk_accuracies(preds, new_labels, k)
    worst_item = min(classwise_topk_acc.items(), key=lambda x: x[1])
    return worst_item[1]


def worst_balanced_n_classes_accuracy(
    classwise_accuracies_dict: dict, n: int
) -> np.array:
    """
    Computes the balanced accuracy among the worst n classes, based on their
    per-class accuracies.

    Args:
        classwise_accuracies_dict: computed accuracies per each class.
        n: number of predicted classes at the bottom of the ranking.
    """
    sorted_classwise_accuracies = sorted(
        classwise_accuracies_dict.items(), key=lambda item: item[1]
    )
    n_worst = sorted_classwise_accuracies[:n]
    return np.array([x[1] for x in n_worst]).mean()


def worst_heuristic_n_classes_recall(preds, new_labels, n) -> np.float64:
    """
    Computes recall for n worst in terms of their per class accuracy.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
         Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        n: number of predicted classes at the bottom of the ranking.
    """
    classwise_accuracies_dict = classwise_accuracies(preds, new_labels)
    classwise_accuracies_sample_numbers = classwise_sample_numbers(new_labels)
    sorted_classwise_accuracies = sorted(
        classwise_accuracies_dict.items(), key=lambda item: item[1]
    )
    n_worst = sorted_classwise_accuracies[:n]
    n_worstclass_recall = (
        np.array([v * classwise_accuracies_sample_numbers[c] for c, v in n_worst]).sum()
        / np.array([classwise_accuracies_sample_numbers[c] for c, v in n_worst]).sum()
    )
    return n_worstclass_recall


def worst_balanced_n_classes_topk_accuracy(preds, new_labels, n, k) -> np.float64:
    """
    Computes the balanced accuracy for the worst n classes in therms of their per class topk accuracy

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        n: number of predicted classes at the bottom of the ranking.
        k: number of predicted classes at the top of the ranking that used in
            topk accuracy.
    """
    classwise_topk_accuracies_dict = classwise_topk_accuracies(preds, new_labels, k)
    sorted_clw_topk_acc = sorted(
        classwise_topk_accuracies_dict.items(), key=lambda item: item[1]
    )
    n_worst = sorted_clw_topk_acc[:n]
    return np.array([x[1] for x in n_worst]).mean()


def worst_heuristic_n_classes_topk_recall(preds, new_labels, n, k) -> np.float64:
    """
    Computes the recall for the worst n classes in therms of their per class topk accuracy.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        n: number of predicted classes at the bottom of the ranking.
        k: number of predicted classes at the top of the ranking that used in
            topk accuracy.
    """
    classwise_topk_accuracies_dict = classwise_topk_accuracies(preds, new_labels, k)
    classwise_accuracies_sample_numbers = classwise_sample_numbers(new_labels)
    sorted_clw_topk_acc = sorted(
        classwise_topk_accuracies_dict.items(), key=lambda item: item[1]
    )
    n_worst = sorted_clw_topk_acc[:n]
    n_worstclass_recall = (
        np.array([v * classwise_accuracies_sample_numbers[c] for c, v in n_worst]).sum()
        / np.array([classwise_accuracies_sample_numbers[c] for c, v in n_worst]).sum()
    )
    return n_worstclass_recall


def worst_balanced_superclass_recall(
    classwise_accuracies_dict, superclasses
) -> np.float64:
    """
    Computes the worst balanced recall among the superclasses.

    Args:
        classwise_accuracies_dict: computed accuracies per each class.
        superclasses: output of worst_case.WorstCase.superclasses.
    """
    superclass_classwise_accuracies = {
        i: np.array([classwise_accuracies_dict[c] for c in s]).mean()
        for i, s in enumerate(superclasses)
    }
    worst_item = min(superclass_classwise_accuracies.items(), key=lambda x: x[1])
    return worst_item[1]


def worst_superclass_recall(preds, new_labels, superclasses) -> np.float:
    """
    Computes the worst not balanced recall among the superclasses.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        superclasses: output of worst_case.WorstCase.superclasses.
    """
    classwise_accuracies_dict = classwise_accuracies(preds, new_labels)
    classwise_sample_number = classwise_sample_numbers(new_labels)
    superclass_classwise_accuracies = {
        i: np.array(
            [classwise_accuracies_dict[c] * classwise_sample_number[c] for c in s]
        ).sum()
        / np.array([classwise_sample_number[c] for c in s]).sum()
        for i, s in enumerate(superclasses)
    }
    worst_item = min(superclass_classwise_accuracies.items(), key=lambda x: x[1])
    return worst_item[1]


def worst_class_precision(preds, new_labels) -> np.float:
    """
    Computes the precision for the worst class.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.

    Returns:
       Dict entry with the worst performing class.
    """
    classes = list(set(new_labels))
    per_class_precision = {}
    for c in classes:
        erroneous_c = (preds["predicted_classes"] == c) * (new_labels != c)
        correct_c = (preds["predicted_classes"] == c) * (new_labels == c)
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


def class_confusion(preds, new_labels) -> np.array:
    """Computes the confusion matrix.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.

    Returns:
        Confusion matrix.
    """
    classes = list(set(new_labels))
    confusion = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(new_labels):
        confusion[c, preds["predicted_classes"][i]] += 1
    return confusion


def intra_superclass_accuracies(probs, new_labels, superclasses) -> dict:
    """
    Computes the accuracy for the images among one superclass, for each superclass.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        superclasses: output of worst_case.WorstCase.superclasses.
    """
    intra_superclass_accuracies = {}
    original_probs = probs.copy()
    original_targets = new_labels.copy()
    for i, s in enumerate(superclasses):
        probs = original_probs.copy()
        new_labels = original_targets.copy()

        internal_samples = np.isin(new_labels, s)
        internal_targets = new_labels[internal_samples]
        internal_probs = probs[internal_samples][:, s]
        s_targets = np.vectorize(lambda x: s[x])
        probs = internal_probs
        internal_preds = s_targets(probs.argmax(axis=1))
        intra_superclass_accuracies[i] = (internal_preds == internal_targets).mean()
    return intra_superclass_accuracies


def worst_intra_superclass_accuracy(probs, new_labels, superclasses) -> np.float64:
    """
    Computes the worst superclass accuracy using intra_superclass_accuracies.

    Args:
        preds: output of worst_case.WorstCase.get_predictions().
            Predictions and probabilities for the used model.
        new_labels: cleaned labels, worst_case.WorstCase.new_labels property.
        superclasses: output of worst_case.WorstCase.superclasses.

    Returns:
        The accuracy for the worst super class.
    """
    isa = intra_superclass_accuracies(probs, new_labels, superclasses)
    worst_item = min(isa.items(), key=lambda x: x[1])
    return worst_item[1]
