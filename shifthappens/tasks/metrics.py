"""Categories in which metrics of tasks will be grouped."""

import enum


class Metric(enum.IntEnum):
    """Categories which will be used to aggregate and average over  the results of various tasks in the benchmark."""

    Calibration = enum.auto()
    Robustness = enum.auto()
    Fairness = enum.auto()
    OODDetection = enum.auto()
    Consistency = enum.auto()
    Miscellaneous = enum.auto()
