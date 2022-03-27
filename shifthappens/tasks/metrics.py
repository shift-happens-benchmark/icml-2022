"""Categories in which metrics of tasks will be grouped."""

import enum


class Metric(enum.IntEnum):
    Calibration = enum.auto()
    Robustness = enum.auto()
    Fairness = enum.auto()
    OODDetection = enum.auto()
    Consistency = enum.auto()
    Miscellaneous = enum.auto()
