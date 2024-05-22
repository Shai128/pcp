from typing import List

import numpy as np
import torch
import tqdm

from calibration_schemes.AbstractCalibration import Calibration
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


class CQRCalibration(Calibration):

    def __init__(self, alpha: float, ignore_masked=False):
        super().__init__(alpha)
        self.Qs = None
        self.ignore_masked = ignore_masked

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: PredictionIntervals, **kwargs):
        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        cal_uncalibrated_intervals = cal_prediction.intervals
        if len(y_cal.shape) == 1:
            y_cal = y_cal.unsqueeze(-1)
        if len(cal_uncalibrated_intervals.shape) == 2:
            cal_uncalibrated_intervals = cal_uncalibrated_intervals.unsqueeze(1)
        corrected_alpha = get_corrected_alpha(self.alpha, y_cal.shape[-1])
        if self.ignore_masked:
            y_cal, deleted_cal, cal_uncalibrated_intervals = y_cal[~deleted_cal], deleted_cal[~deleted_cal],\
                                                             cal_uncalibrated_intervals[~deleted_cal]
        self.Qs = multi_dimensional_cqr_calibration(y_cal, deleted_cal, cal_uncalibrated_intervals, corrected_alpha)
        self.Qs = self.Qs.squeeze()

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_uncalibrated_intervals: PredictionIntervals,
                                              **kwargs) -> PredictionIntervals:
        test_uncalibrated_intervals = test_uncalibrated_intervals.intervals.clone()
        test_uncalibrated_intervals[..., 0] -= self.Qs
        test_uncalibrated_intervals[..., 1] += self.Qs
        return PredictionIntervals(test_uncalibrated_intervals)

    def compute_scores(self, x, y, test_uncalibrated_intervals: PredictionIntervals):
        if len(y.shape) == 2:
            y = y.squeeze(1)
        assert len(y.shape) == 1
        return compute_cqr_scores(y, test_uncalibrated_intervals.intervals)

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> PredictionIntervals:
        if not isinstance(test_prediction, PredictionIntervals):
            raise Exception(f"test_prediction must be of type PredictionIntervals for CQR calibration,"
                            f" but found {type(test_prediction)}")
        test_uncalibrated_intervals = test_prediction.intervals.clone()
        if isinstance(threshold, List):
            threshold = torch.Tensor(threshold).to(test_uncalibrated_intervals.device)
        elif isinstance(threshold, np.ndarray):
            threshold = torch.Tensor(threshold).to(test_uncalibrated_intervals.device)
        test_uncalibrated_intervals[..., 0] -= threshold
        test_uncalibrated_intervals[..., 1] += threshold
        return PredictionIntervals(test_uncalibrated_intervals)

    @property
    def name(self):
        if self.ignore_masked:
            return 'cqr_ignore_masked'
        else:
            return "cqr"

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal,
                                                             cal_predictions: List[ModelPrediction],
                                                             x_test,
                                                             test_prediction: List[ModelPrediction],
                                                             cal_weights=None,
                                                             test_weights=None,
                                                             **kwargs) -> UncertaintySets:
        assert isinstance(cal_predictions[0], PredictionIntervals)
        assert isinstance(test_prediction[0], PredictionIntervals)
        cal_intervals = torch.stack([cp.intervals for cp in cal_predictions]).squeeze(1)
        if self.ignore_masked:
            cal_intervals = cal_intervals[~deleted_cal]
            y_cal = y_cal[~deleted_cal]
            test_prediction = [test_prediction[i] for i in range(len(test_prediction)) if (~deleted_cal)[i].item()]
        device = x_cal.device
        # assert isinstance(test_prediction, List[PredictionIntervals])
        # cal_intervals = cal_prediction.intervals
        cal_upper_error = y_cal.squeeze() - cal_intervals[:, 1].squeeze()
        cal_lower_error = cal_intervals[:, 0].squeeze() - y_cal.squeeze()
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = self.alpha
        corrected_alpha = get_corrected_alpha(alpha, y_cal.shape[-1])
        # alpha_low = corrected_alpha - 1 / (y_cal.shape[0] + 1)
        alpha_high = 1-corrected_alpha + 1 / (y_cal.shape[0] + 1)
        calibrated_intervals = []

        for i in tqdm.tqdm(range(len(x_test))):

            if cal_weights is not None and test_weights is not None:
                w_test_i = test_weights[i]
                p_i = cal_weights / (cal_weights.sum() + w_test_i)
                p_test = w_test_i / (cal_weights.sum() + w_test_i)
                sample_weight = torch.cat([p_i, torch.tensor([p_test.item()], device=device)])
            else:
                sample_weight = None
            curr_predictions = torch.stack([pred[i].intervals for pred in test_prediction]).clone()
            curr_predictions[:, 1] += cal_upper_error
            curr_predictions[:, 0] -= cal_lower_error

            upper_values = torch.cat([curr_predictions[:, 1], torch.tensor([curr_predictions[:, 1].max()], device=device)])
            lower_values = torch.cat([-curr_predictions[:, 0], torch.tensor([curr_predictions[:, 0].min()], device=device)])
            # upper_q = torch.quantile(curr_predictions[:, 1], q=alpha_high).item()
            # lower_q = torch.quantile(curr_predictions[:, 0], q=alpha_low).item()

            upper_q = weighted_quantile(upper_values, alpha_high, sample_weight=sample_weight, old_style=False).item()
            lower_q = -weighted_quantile(lower_values, alpha_high, sample_weight=sample_weight, old_style=False).item()
            calibrated_intervals += [torch.Tensor([lower_q, upper_q]).to(device).unsqueeze(-1)]

        calibrated_intervals = torch.stack(calibrated_intervals, dim=0).squeeze(-1)
        return PredictionIntervals(calibrated_intervals)