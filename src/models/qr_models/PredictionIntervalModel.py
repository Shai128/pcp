from __future__ import annotations

import abc
from typing import List

import torch
from models.abstract_models.AbstractModel import Model
from models.model_utils import UncertaintySets, ModelPrediction


class PredictionIntervals(UncertaintySets, ModelPrediction):

    def __init__(self, intervals: torch.Tensor):
        super().__init__()
        self.intervals = intervals
        if len(intervals.shape) == 2:
            self.intervals = self.intervals.squeeze(1)
    def __getitem__(self, item):
        return PredictionIntervals(self.intervals.__getitem__(item))

    def contains(self, y):
        y = y.squeeze()
        is_in_interval = (y <= self.intervals[..., 1]) & (y >= self.intervals[..., 0])
        if len(is_in_interval.shape) == 2:
            is_in_interval_result = torch.ones(len(y)).to(y.device).bool()
            for k in range(is_in_interval.shape[-1]):
                is_in_interval_result &= is_in_interval[:, k]
            return is_in_interval_result
        else:
            return is_in_interval

    def union(self, other_set: UncertaintySets) -> PredictionIntervals:
        assert isinstance(other_set, PredictionIntervals)
        intervals1 = self.intervals
        intervals2 = other_set.intervals
        min_vals = torch.min(intervals1[..., 0], intervals2[..., 0])
        max_vals = torch.max(intervals1[..., 1], intervals2[..., 1])
        return PredictionIntervals(torch.cat([min_vals.unsqueeze(-1), max_vals.unsqueeze(-1)], dim=-1))


class PredictionIntervalModel(Model):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    @abc.abstractmethod
    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        pass

    @abc.abstractmethod
    def construct_uncalibrated_intervals(self, x: torch.Tensor) -> PredictionIntervals:
        pass

    @abc.abstractmethod
    def eval(self):
        pass
