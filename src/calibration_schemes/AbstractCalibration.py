import abc
from typing import List

import torch

from models.model_utils import UncertaintySets, ModelPrediction


class Calibration(abc.ABC):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.calibrated_count = 0
        self.fit_count = 0

    @abc.abstractmethod
    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ModelPrediction, **kwargs):
        if self.fit_count == 0:
            print(f"warning: {self.name} calibration is calibrated without being fit")

        if self.calibrated_count > 0:
            print(f"warning: {self.name} calibration was calibrated already calibrated {self.calibrated_count} times and is now called again")
        self.calibrated_count += 1

    @abc.abstractmethod
    def compute_scores(self, x, y, cal_prediction: ModelPrediction):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        if self.fit_count > 0:
            print(f"warning: {self.name} calibration was fitted already {self.fit_count} times and is now called again")
        self.fit_count += 1

    @abc.abstractmethod
    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ModelPrediction, **kwargs) -> UncertaintySets:
        pass

    @abc.abstractmethod
    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> UncertaintySets:
        pass

    @abc.abstractmethod
    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test,
                                                             model_predictions: List[ModelPrediction], **kwargs) -> UncertaintySets:
        pass

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test, test_model_prediction: ModelPrediction) -> dict:
        return {}
