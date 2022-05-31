"""Utils for metrics implementations for calculating models performance."""

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_ood(values_in: np.ndarray, values_out: np.ndarray) -> float:
    """
    Implementation of Area-under-Curve metric for out-of-distribution detection.
    The higher the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.

    Returns:
        Area-under-curve score.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    y_true = len(values_in) * [1] + len(values_out) * [0]
    y_score = np.nan_to_num(np.concatenate([values_in, values_out]).flatten())
    return roc_auc_score(y_true, y_score)


def fpr_at_tpr(values_in: np.ndarray, values_out: np.ndarray, tpr: float) -> float:
    """
    Implementation of FPR metric at the particular TPR for out-of-distribution detection.
    The lower the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.
        tpr: (1 - true positive rate), for which probability threshold is calculated for
            in-domain data.

    Returns:
        False positive rate on out-of-domain data at (1-tpr) threshold.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    fpr = (values_out >= t).mean()
    return fpr
