from typing import List

import torch

from calibration_schemes.AbstractCalibration import Calibration
from models.qr_models.PredictionIntervalModel import PredictionIntervals
from models.model_utils import ModelPrediction, UncertaintySets


class DummyCalibration(Calibration):

    def __init__(self, alpha: float):
        super().__init__(alpha)
        self.Qs = None

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: PredictionIntervals, **kwargs):
        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        pass

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor, test_prediction: ModelPrediction,
                                              **kwargs) -> UncertaintySets:
        if not isinstance(test_prediction, UncertaintySets):
            raise Exception("test_prediction must be of type test_prediction for dummy calibration")
        return test_prediction

    @property
    def name(self):
        return "Dummy"

    def compute_scores(self, x, y, cal_prediction: ModelPrediction):
        return torch.zeros(len(y)).to(y.device)

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> UncertaintySets:
        raise NotImplementedError()

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test, test_prediction: List[ModelPrediction], **kwargs) -> UncertaintySets:
        if not isinstance(test_prediction[0], UncertaintySets):
            raise Exception("test_prediction must be of type test_prediction for dummy calibration")
        return test_prediction[0]
