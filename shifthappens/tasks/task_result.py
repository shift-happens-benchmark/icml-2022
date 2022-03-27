from typing import Dict, Tuple, Union
from .metrics import Metric


class TaskResult:
    """Contains the results of a result, which can be arbitrary metrics.
    At least one of these metrics must be references as a summary metric."""

    def __init__(self, *, summary_metrics: Dict[Metric, Union[str, Tuple[str, ...]]],
                 **metrics: Union[float, int]):
        # validate that metrics referenced in summary metrics exist
        for sm in summary_metrics:
            assert isinstance(sm, Metric), "Invalid summary metric key."
            if isinstance(summary_metrics[sm], str):
                tms = (summary_metrics[sm],)
            else:
                tms = summary_metrics[sm]
            for tm in tms:
                assert tm in metrics

        self._metrics = metrics
        self.summary_metrics = summary_metrics
