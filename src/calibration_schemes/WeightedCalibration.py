from typing import List

import numpy as np
import torch
import tqdm

from calibration_schemes.AbstractCalibration import Calibration
from data_utils.data_scaler import DataScaler
from models.qr_models.PredictionIntervalModel import PredictionIntervalModel
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.model_utils import ModelPrediction, UncertaintySets
from utils import weighted_quantile


class WeightedCalibration(Calibration):
    """
    Implementation of weighted conformal prediction (https://arxiv.org/abs/1904.06019)
    The weights are computed using X,Z. Requires Z to be observed during inference time.
    """

    def __init__(self, base_y_calibration: Calibration, alpha: float,
                 dataset_name: str, data_scaler: DataScaler,
                 data_mask_estimator: DataMaskEstimator):
        super().__init__(alpha)
        self.Qs = None
        self.data_mask_estimator = data_mask_estimator
        self.dataset_name = dataset_name
        self.data_scaler = data_scaler
        self.base_y_calibration = base_y_calibration
        self.marginal_missing_probability = None
        self.weights = None
        self.y2_cal_scores = None

    def get_mask_probabilities(self, scaled_x, scaled_z):
        return

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20,
            **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
                    batch_size=batch_size, n_wait=batch_size, **kwargs)
        self.base_y_calibration.fit(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs,
                                    batch_size,
                                    n_wait, **kwargs)
        self.data_mask_estimator.fit(x_train, z_train, deleted_train, x_val, z_val, deleted_val, epochs=epochs,
                                     batch_size=batch_size, n_wait=n_wait, **kwargs)

    def get_weight(self, conditional_missing_probability):
        return (1 - self.marginal_missing_probability) / (1 - conditional_missing_probability)

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ModelPrediction, **kwargs):
        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        self.data_mask_estimator.calibrate(x_cal, z_cal, deleted_cal)
        conditional_missing_probability = self.data_mask_estimator.predict(x_cal, z_cal)
        self.marginal_missing_probability = conditional_missing_probability.mean()
        self.weights = self.get_weight(conditional_missing_probability)[~deleted_cal]
        self.y2_cal_scores = self.base_y_calibration.compute_scores(x_cal, y_cal, cal_prediction)[~deleted_cal].detach()

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ModelPrediction, **kwargs) -> UncertaintySets:
        if 'z_test' not in kwargs:
            raise Exception(f"could not calibrate with method {self.name} without 'z_test'")
        z_test = kwargs['z_test']
        thresholds = []
        missing_probabilities = self.data_mask_estimator.predict(x_test, z_test)
        device = x_test.device
        max_score = self.y2_cal_scores.max().item()
        quantiles = torch.Tensor([1 - self.alpha]).to(x_test.device)
        for i in tqdm.tqdm(range(len(x_test))):
            missing_probability = missing_probabilities[i]
            w_test = self.get_weight(missing_probability)
            p_i = self.weights / (self.weights.sum() + w_test)
            p_test = w_test / (self.weights.sum() + w_test)
            sample_weight = torch.cat([p_i, torch.tensor([p_test.item()], device=device)])
            values = torch.cat([self.y2_cal_scores, torch.tensor([max_score], device=device)])
            Q = weighted_quantile(values, quantiles, sample_weight=sample_weight, old_style=False).item()
            thresholds += [Q]
        thresholds = torch.Tensor(thresholds).to(x_test.device)
        test_calibrated_sets = self.base_y_calibration.compute_uncertainty_set_from_prediction_and_threshold(
            test_prediction, thresholds)

        return test_calibrated_sets

    def compute_scores(self, x, y, cal_prediction: ModelPrediction):
        return self.base_y_calibration.compute_scores(x, y, cal_prediction)

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> UncertaintySets:
        raise NotImplementedError("not implemented yet")

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal,
                                                             cal_predictions: List[ModelPrediction],
                                                             x_test,
                                                             test_prediction: List[ModelPrediction],
                                                             z_test=None, **kwargs) -> UncertaintySets:
        if z_test is None:
            raise Exception(f"{self.name} must get z_test, but got {z_test}")
        missing_probabilities = self.data_mask_estimator.predict(x_cal, z_cal)
        marginal_missing_probability = missing_probabilities.mean()
        cal_weights = (1 - marginal_missing_probability) / (1 - missing_probabilities)[~deleted_cal]
        new_deleted_cal = torch.zeros(len(cal_weights)).to(cal_weights.device)
        test_missing_probabilities = self.data_mask_estimator.predict(x_test, z_test)
        test_weights = (1 - marginal_missing_probability) / (1 - test_missing_probabilities)
        cal_predictions = [cal_predictions[i] for i in (~deleted_cal).nonzero()]
        test_prediction = [test_prediction[i] for i in (~deleted_cal).nonzero()]
        return self.base_y_calibration.jackknife_plus_construct_uncertainty_set_from_scores(x_cal[~deleted_cal],
                                                                                            y_cal[~deleted_cal],
                                                                                            z_cal[~deleted_cal],
                                                                                            new_deleted_cal,
                                                                                            cal_predictions,
                                                                                            x_test,
                                                                                            test_prediction,
                                                                                            cal_weights=cal_weights,
                                                                                            test_weights=test_weights,
                                                                                            )

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test,
                            test_model_prediction: ModelPrediction) -> dict:
        return {
            **self.data_mask_estimator.compute_performance(x_test, z_test, full_y_test, deleted_test),
            **self.base_y_calibration.compute_performance(x_test, y, z_test, full_y_test, deleted_test,
                                                          test_model_prediction)
        }

    @property
    def name(self):
        return f"weighted_{self.base_y_calibration.name}_{self.data_mask_estimator.name}_masker"
