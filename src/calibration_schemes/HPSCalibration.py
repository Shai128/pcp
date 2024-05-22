from typing import List
import numpy as np
import torch

from calibration_schemes.AbstractCalibration import Calibration
from models.ClassificationModel import PredictionSets, ClassProbabilities
from models.model_utils import ModelPrediction, UncertaintySets


class HPSCalibration(Calibration):

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
        cal_prediction = cal_prediction.probabilities
        n = y_cal.shape[0]
        scores = 1 - cal_prediction[range(n), y_cal.round().long().squeeze()]
        assert len(scores.shape) == 1
        quantile_level = min(0.9999, 1 - self.alpha + (1 / (n + 1)))
        self.q = torch.quantile(scores, q=quantile_level)

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ClassProbabilities, **kwargs) -> PredictionSets:
        scores: torch.Tensor = 1 - test_prediction.probabilities
        prediction_set = scores <= self.q
        return PredictionSets(prediction_set)

    @property
    def name(self):
        if self.ignore_masked:
            return "hps_ignore_masked"
        else:
            return "hps"

    def compute_scores(self, x, y, cal_prediction: ClassProbabilities):
        y = y.squeeze(1)
        n = y.shape[0]
        cal_prediction = cal_prediction.probabilities
        scores = 1 - cal_prediction[range(n), y.round().long()]
        return scores

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ClassProbabilities,
                                                              threshold) -> UncertaintySets:
        scores: torch.Tensor = 1 - test_prediction.probabilities
        if torch.is_tensor(threshold) and len(threshold.shape) == 1:
            threshold = threshold.unsqueeze(-1)
            threshold = threshold.repeat(1, scores.shape[-1])
        if isinstance(threshold, List):
            threshold = torch.Tensor(threshold).unsqueeze(1).to(scores.device)
        prediction_set = scores <= threshold
        return PredictionSets(prediction_set)

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test,
                                                             model_predictions: List[ModelPrediction],
                                                             **kwargs) -> UncertaintySets:
        raise NotImplementedError()

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test,
                            test_model_prediction: ModelPrediction) -> dict:
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
