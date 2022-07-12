"""Evaluates a collection of models on all tasks of the benchmark and calculates the ranking-scores
of the models for every metric (i.e. type of task)."""

from typing import Dict

from shifthappens.tasks.metrics import Metric


class ScoreCard:
    """Contains the ranking scores of a model relative to all other models.

    Args:
        relative_performance: A dict where the keys are `shifthappens.tasks.metrics.Metric`
            items and the values are the average quantile of the model's performance.
    """
    def __init__(self, relative_performance: Dict[Metric, float]):
        self.relative_performance = relative_performance

    @property
    def summary(self):
        """A summary of the final ranking scores of the model."""
        return self.relative_performance.copy()


def evaluate_all_models():
    """Runs the benchmark for all registered models and saves their scores."""
    raise NotImplementedError()


def score_models():
    """Calculates the relative performance for all models that have previously been
    evaluated using ``evaluate_all_models``"""
    raise NotImplementedError()
