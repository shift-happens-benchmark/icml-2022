"""Class for representing the results of a single task."""

from typing import Dict
from typing import Tuple
from typing import Union

from .metrics import Metric


class TaskResult:
    """Contains the results of a result, which can be arbitrary metrics.
    At least one of these metrics must be references as a summary metric.

    Args:
        summary_metrics: Associates :py:class:`shifthappens.tasks.metrics.Metric` values
            to the name of metrics calculated by the task.
        metrics: Metrics' names and their values.
    Examples:
        >>> @dataclasses.dataclass
        >>> class CustomTask(Task):
        >>>     ...
        >>>     def _evaluate(self, model: shifthappens.models.base.Model) -> DataLoader:
        >>>         ...
        >>>         return TaskResult(
        >>>                 your_robustness_metric=your_robustness_metric,
        >>>                 your_calibration_metric=your_calibration_metric,
        >>>                 your_custom_metric=1.0 - your_custom_metric,
        >>>                 summary_metrics={
        >>>                    Metric.Robustness: "your_robustness_metric",
        >>>                    Metric.Calibration: "your_calibration_metric"},
        >>>                 )
        >>>     ...
    """

    def __init__(
        self,
        *,
        summary_metrics: Dict[Metric, Union[str, Tuple[str, ...]]],
        **metrics: Union[float, int],
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

    def serialize_summary_metrics(self) -> str:
        """
        Serializes summary metrics of the objects into a string.
        """
        return str({key.name:value for (key, value) in self.summary_metrics.items()})


    def serialize_task_result(self) -> str:
        """
        Serializes TaskResult object into a string.
        """
        result_dict = {
            'summary_metrics': self.serialize_summary_metrics(),
            'metrics': str(self._metrics)
        }
        return str(result_dict)

    @staticmethod
    def deserialize_summary_metrics(summary_metrics_str:str) -> Dict[Metric, Union[str, Tuple[str, ...]]]:
        """
        Deserializes valid string into summary_metrics.
        """
        summary_metrics = eval(summary_metrics_str)
        result = {}
        for (key,value) in summary_metrics.items():
            result[Metric.__members__.get(key)] = value
        return result

    @staticmethod
    def deserialize_task_result(task_result_str: str):
        """
        Deserializes valid string into a TaskResult object.
        """
        result_dict = eval(task_result_str)
        metrics = eval(result_dict['metrics'])
        summary_metrics = TaskResult.deserialize_summary_metrics(result_dict['summary_metrics'])
        return TaskResult(
            summary_metrics=summary_metrics,
            **metrics
        )
