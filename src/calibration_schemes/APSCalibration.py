from typing import List

import numpy as np
from scipy.stats import rankdata

import torch

from calibration_schemes.AbstractCalibration import Calibration
from models.ClassificationModel import PredictionSets, ClassProbabilities
from models.model_utils import ModelPrediction, UncertaintySets


class APSCalibration(Calibration):

    def __init__(self, alpha: float, ignore_masked=False):
        super().__init__(alpha)
        self.q = None
        self.ignore_masked = ignore_masked

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ClassProbabilities, **kwargs):
        if deleted_cal is None:
            deleted_cal = torch.zeros(y_cal.shape[0], device=y_cal.device, dtype=torch.bool)

        if self.ignore_masked:
            x_cal = x_cal[~deleted_cal]
            y_cal = y_cal[~deleted_cal]
            z_cal = z_cal[~deleted_cal]
            cal_prediction = cal_prediction[~deleted_cal]
            deleted_cal = deleted_cal[~deleted_cal]

        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        cal_scores = self.compute_scores(x_cal, y_cal, cal_prediction)
        n = len(cal_scores)
        self.q = np.quantile(cal_scores, np.ceil((n + 1) * (1 - self.alpha)) / n, interpolation='higher').item()

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ClassProbabilities, **kwargs) -> PredictionSets:
        test_probabilities = test_prediction.probabilities.cpu().detach().numpy()
        val_pi = test_probabilities.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(test_probabilities, val_pi, axis=1).cumsum(axis=1)
        prediction_sets = np.take_along_axis(val_srt <= self.q, val_pi.argsort(axis=1), axis=1)
        prediction_sets = torch.Tensor(prediction_sets).to(x_test.device)
        return PredictionSets(prediction_sets)

    @property
    def name(self):
        if self.ignore_masked:
            return "aps_ignore_masked"
        else:
            return "aps"

    def compute_scores(self, x, y, cal_prediction: ClassProbabilities) -> torch.Tensor:
        probabilities = cal_prediction.probabilities.cpu().numpy()
        n = len(x)
        # x = cal_prediction.probabilities.cpu().numpy()
        y = y.cpu().numpy()
        cal_pi = probabilities.argsort(axis=1)[:, ::-1]
        cal_srt = np.take_along_axis(probabilities, cal_pi, axis=1).cumsum(axis=1)
        cal_scores = np.take_along_axis(cal_srt, cal_pi.argsort(axis=1), axis=1)[
            list(range(n)), y.astype(np.int32).squeeze()]

        return cal_scores

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ClassProbabilities,
                                                              threshold) -> UncertaintySets:
        raise NotImplementedError()

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test,
                                                             model_predictions: List[ModelPrediction],
                                                             **kwargs) -> UncertaintySets:
        raise NotImplementedError()

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test, test_model_prediction: ModelPrediction) -> dict:
        performance = super().compute_performance(x_test, y, z_test, full_y_test, deleted_test, test_model_prediction)
        q = self.q
        if torch.is_tensor(q):
            q = q.item()
        if isinstance(q, np.ndarray):
            q = q.item()
        return {
            **performance,
            'Q': q
        }