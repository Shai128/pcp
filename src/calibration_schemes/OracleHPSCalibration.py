from typing import List

import torch

from calibration_schemes.AbstractCalibration import Calibration
from calibration_schemes.HPSCalibration import HPSCalibration
from models.ClassificationModel import PredictionSets, ClassProbabilities
from models.model_utils import ModelPrediction, UncertaintySets


class OracleHPSCalibration(HPSCalibration):

    def __init__(self, alpha: float):
        super().__init__(alpha, ignore_masked=False)
        self.q = None

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ClassProbabilities, **kwargs):
        assert 'full_y_cal' in kwargs
        full_y_cal = kwargs['full_y_cal']
        deleted_cal = None
        super().calibrate(x_cal, full_y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)

    @property
    def name(self):
        return "oracle_hps"

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test,
                                                             model_predictions: List[ModelPrediction], **kwargs) -> UncertaintySets:
        raise NotImplementedError()
