import dataclasses

import pytest

import shifthappens.benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.tasks.base import parameter
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


def test_iterate_flavours():
    @dataclasses.dataclass
    class DummyTask(Task):
        a: int = parameter(default=0, options=(0, 1))
        b: int = parameter(default=2, options=(2, 3))
        c: int = parameter(default=4, options=(4, 5))

        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> TaskResult:
            pass

        def _prepare(self, model: sh_models.Model) -> DataLoader:
            pass

        def _prepare_data(self) -> DataLoader:
            pass

    flavours = list(DummyTask.iterate_flavours(data_root="test"))
    assert len(flavours) == 2 * 2 * 2
    for flavour in flavours:
        assert isinstance(flavour.a, int)
        assert isinstance(flavour.b, int)
        assert isinstance(flavour.c, int)


def test_register_unregister_task():
    n_previous_registered_tasks = len(sh_benchmark.get_registered_tasks())

    # check whether registration works
    @sh_benchmark.register_task(name="dummy_task", relative_data_folder="dummy_task")
    @dataclasses.dataclass
    class DummyTask(Task):
        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> TaskResult:
            pass

        def _prepare(self, model: sh_models.Model) -> DataLoader:
            pass

        def _prepare_data(self) -> DataLoader:
            pass

    assert len(sh_benchmark.get_registered_tasks()) == n_previous_registered_tasks + 1
    assert DummyTask in sh_benchmark.get_registered_tasks()

    # check whether unregistration works
    sh_benchmark.unregister_task(DummyTask)
    assert len(sh_benchmark.get_registered_tasks()) == n_previous_registered_tasks
    assert DummyTask not in sh_benchmark.get_registered_tasks()

    # check whether unregistration only works for registered tasks
    with pytest.raises(ValueError):
        sh_benchmark.unregister_task(DummyTask)


def test_data_folder():
    @dataclasses.dataclass
    class DummyTask(Task):
        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> TaskResult:
            pass

        def _prepare(self, model: sh_models.Model) -> DataLoader:
            pass

        def _prepare_data(self) -> DataLoader:
            pass


def test_task_result():
    with pytest.raises(TypeError):
        TaskResult(accuracy=1, error=0, confidence=0)

    with pytest.raises(AssertionError):
        TaskResult(
            summary_metrics=dict(a="accuracy"), accuracy=1, error=0, confidence=0
        )

    TaskResult(
        summary_metrics={Metric.Robustness: "accuracy"},
        accuracy=1,
        error=0,
        confidence=0,
    )
