from typing import Dict
from typing import Tuple
from typing import Union

from .metrics import Metric


class TaskResult:
    """Contains the results of a result, which can be arbitrary metrics.
    At least one of these metrics must be references as a summary metric.

    Args:
        summary_metrics (dict): Associates ``shifthappens.tasks.metrics.Metric``s
            to the name of metrics calculated by the task.
        **metrics (float, int): Metrics' names and their values.
    """

    def __init__(
        self,
        *,
        summary_metrics: Dict[Metric, Union[str, Tuple[str, ...]]],
        **metrics: float,
    ):
        # validate that metrics referenced in summary metrics exist
        for sm in summary_metrics:
            assert isinstance(sm, Metric), "Invalid summary metric key."
            smv = summary_metrics[sm]
            if isinstance(smv, str):
                tms: Tuple[str, ...] = (smv,)
            elif isinstance(smv, tuple):
                tms = smv
            else:
                raise ValueError(
                    f"Value for metric key `{sm}` is neither str nor tuple of str."
                )
            for tm in tms:
                assert tm in metrics

        self._metrics = metrics
        self.summary_metrics = summary_metrics

    def __getitem__(self, item) -> float:
        return self._metrics[item]

    def __getattr__(self, item) -> float:
        if item in self._metrics:
            return self[item]
        else:
            return super().__getattribute__(item)
