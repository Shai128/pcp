from typing import List

import numpy as np
import torch
import tqdm

from calibration_schemes.AbstractCalibration import Calibration
from calibration_schemes.CQRCalibration import CQRCalibration
from calibration_schemes.calibration_utils import multi_dimensional_cqr_calibration, compute_cqr_scores
from models.model_utils import ModelPrediction, UncertaintySets
from models.qr_models.PredictionIntervalModel import PredictionIntervals
from utils import weighted_quantile


def get_corrected_alpha(alpha, dim):
    if dim > 1:
        corrected_alpha = 1 - ((1 - alpha) ** (1 / dim))
    else:
        corrected_alpha = alpha
    return corrected_alpha


class OracleCQRCalibration(Calibration):

    def __init__(self, alpha: float):
        super().__init__(alpha)
        self.cqr_calibration = CQRCalibration(alpha)

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: PredictionIntervals,
                  **kwargs):
        assert 'full_y_cal' in kwargs
        full_y_cal = kwargs['full_y_cal']
        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        return self.cqr_calibration.calibrate(x_cal, full_y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_uncalibrated_intervals: PredictionIntervals,
                                              **kwargs) -> PredictionIntervals:
        return self.cqr_calibration.construct_calibrated_uncertainty_sets(x_test, test_uncalibrated_intervals, **kwargs)

    def compute_scores(self, x, y, test_uncalibrated_intervals: PredictionIntervals):
        return self.cqr_calibration.compute_performance(x, y, test_uncalibrated_intervals)

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> PredictionIntervals:
        return self.cqr_calibration.compute_uncertainty_set_from_prediction_and_threshold(test_prediction, threshold)

    @property
    def name(self):
        return "oracle_cqr"

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal,
                                                             cal_predictions: List[ModelPrediction],
                                                             x_test,
                                                             test_prediction: List[ModelPrediction],
                                                             cal_weights=None,
                                                             test_weights=None,
                                                             **kwargs) -> UncertaintySets:
        full_y_cal = kwargs['full_y_cal']
        return self.cqr_calibration.jackknife_plus_construct_uncertainty_set_from_scores(
            x_cal, full_y_cal, z_cal, deleted_cal,
            cal_predictions, x_test, test_prediction,
            cal_weights=cal_weights,
            test_weights=test_weights,
            **kwargs
        )
