import dataclasses
from typing import Optional

import shifthappens.benchmark as sh_benchmark
from shifthappens.data.base import DataLoader
from shifthappens.models import base as sh_models
from shifthappens.models.torchvision import ResNet18
from shifthappens.tasks.base import parameter
from shifthappens.tasks.base import Task
from shifthappens.tasks.metrics import Metric
from shifthappens.tasks.task_result import TaskResult


def test_model():
    for task in sh_benchmark.get_registered_tasks():
        sh_benchmark.unregister_task(task)

    @sh_benchmark.register_task(name="dummy_task", relative_data_folder="test")
    @dataclasses.dataclass
    class DummyTask(Task):
        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> TaskResult:
            return TaskResult(
                accuracy=1.0, summary_metrics={Metric.Robustness: "accuracy"}
            )

        def _prepare_dataloader(self) -> Optional[DataLoader]:
            return None

    results = sh_benchmark.evaluate_model(ResNet18(), "test")
    assert len(results) == 1


def test_model_task_flavors():
    for task in sh_benchmark.get_registered_tasks():
        sh_benchmark.unregister_task(task)

    @sh_benchmark.register_task(name="dummy_task", relative_data_folder="test")
    @dataclasses.dataclass
    class DummyTask(Task):
        a: int = parameter(default=0, options=(0, 1))

        def setup(self):
            pass

        def _evaluate(self, model: sh_models.Model) -> TaskResult:
            return TaskResult(
                accuracy=1.0, summary_metrics={Metric.Robustness: "accuracy"}
            )

        def _prepare_dataloader(self) -> Optional[DataLoader]:
            return None

    model = ResNet18()
    results = sh_benchmark.evaluate_model(ResNet18(), "test")
    assert len(results) == 2
