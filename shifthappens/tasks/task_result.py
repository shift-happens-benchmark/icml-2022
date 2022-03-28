from typing import Dict
from typing import Tuple
from typing import Union

from .metrics import Metric


class TaskResult:
    """Contains the results of a result, which can be arbitrary metrics.
    At least one of these metrics must be references as a summary metric."""

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
                tms = (smv,)
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
