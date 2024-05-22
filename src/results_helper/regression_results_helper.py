import torch

from models.qr_models.PredictionIntervalModel import PredictionIntervals
from results_helper.results_helper import ResultsHelper


def compute_coverage(y: torch.Tensor, intervals: torch.Tensor):
    y = y.squeeze()
    intervals = intervals.squeeze()
    assert len(y.shape) == 1 and len(intervals.shape) == 2
    return ((y <= intervals[:, 1]) & (y >= intervals[:, 0])).float().mean().item()


def compute_length(intervals):
    assert len(intervals.shape) == 2
    return (intervals[:, 1] - intervals[:, 0]).mean().item()


class RegressionResultsHelper(ResultsHelper):
    def __init__(self, base_results_save_dir, seed):
        super().__init__(base_results_save_dir, seed)

    def compute_performance_metrics_on_data_aux(self, full_y, y, deleted,
                                                uncalibrated_uncertainty_sets: PredictionIntervals,
                                                calibrated_uncertainty_sets: PredictionIntervals) -> dict:
        full_y = full_y.squeeze()
        y = y.squeeze()
        intervals = calibrated_uncertainty_sets.intervals.squeeze()
        assert len(full_y.shape) == len(y.shape) == len(intervals.shape) - 1 == 1
        y2_coverage = compute_coverage(y, intervals)
        not_deleted_y2_coverage = compute_coverage(y[~deleted], intervals[~deleted])
        deleted_y2_coverage = compute_coverage(full_y[deleted], intervals[deleted])
        full_y2_coverage = compute_coverage(full_y, intervals)
        y2_length = compute_length(intervals)
        return {
            'y2 coverage': y2_coverage,
            '~deleted y2 coverage': not_deleted_y2_coverage,
            'deleted y2 coverage': deleted_y2_coverage,
            'full y2 coverage': full_y2_coverage,
            'y2 length': y2_length
        }

