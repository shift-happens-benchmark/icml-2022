from typing import Dict

from shifthappens.tasks.metrics import Metric


class ScoreCard:
    def __init__(self, relative_performance: Dict[Metric, float]):
        """
        Args:
            relative_performance: A dict where the keys are `shifthappens.tasks.metrics.Metric`
                items and the values are the average quantile of the model's performance.
        """
        self.relative_performance = relative_performance

    @property
    def summary(self):
        return self.relative_performance.copy()


def evaluate_all_models():
    """Runs the benchmark for all registered models and saves their scores."""
    raise NotImplementedError()


def score_models():
    """Calculates the relative performance for all models that have previously been
    evaluated using ``evaluate_all_models``"""
    raise NotImplementedError()
