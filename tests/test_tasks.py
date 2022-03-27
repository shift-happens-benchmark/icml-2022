import pytest
import dataclasses
from typing import Dict

from shifthappens.tasks.base import Task, parameter
from shifthappens.models import base as sh_models
from shifthappens.tasks.task_result import TaskResult
from shifthappens.tasks.metrics import Metric


def test_iterate_flavours():
    @dataclasses.dataclass
    class DummyTask(Task):
        a: int = parameter(default=0, options=(0, 1))
        b: int = parameter(default=2, options=(2, 3))
        c: int = parameter(default=4, options=(4, 5))

        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> Dict[str, float]:
            pass

        def _prepare(self, model: sh_models.Model) -> "DataLoader":
            pass

    flavours = list(DummyTask.iterate_flavours())
    assert len(flavours) == 2*2*2
    for flavour in flavours:
        assert isinstance(flavour.a, int)
        assert isinstance(flavour.b, int)
        assert isinstance(flavour.c, int)


def test_task_result():
    with pytest.raises(TypeError):
        TaskResult(accuracy=1, error=0, confidence=0)

    with pytest.raises(AssertionError):
        TaskResult(summary_metrics=dict(a="accuracy"), accuracy=1, error=0, confidence=0)

    TaskResult(summary_metrics={Metric.Robustness: "accuracy"}, accuracy=1, error=0, confidence=0)
