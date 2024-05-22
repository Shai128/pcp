from typing import List

import numpy as np
import torch
import tqdm

from calibration_schemes.AbstractCalibration import Calibration
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_scaler import DataScaler
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.model_utils import ModelPrediction, UncertaintySets
from utils import weighted_quantile


class PrivilegedConformalPrediction(Calibration):

    def __init__(self, base_y_calibration: Calibration, alpha: float,
                 dataset_name: str, data_scaler: DataScaler, data_mask_estimator: DataMaskEstimator):
        super().__init__(alpha)
        self.data_mask_estimator = data_mask_estimator
        self.dataset_name = dataset_name
        self.data_scaler = data_scaler
        self.base_y_calibration = base_y_calibration
        self.marginal_missing_probability = None
        self.Q = None
        self.beta = 0.005

    def get_mask_probabilities(self, scaled_x, scaled_z):
        pred = self.data_mask_estimator.predict(scaled_x, scaled_z)
        return pred

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
        conditional_missing_probability = self.get_mask_probabilities(x_cal, z_cal)
        self.marginal_missing_probability = conditional_missing_probability.mean().item()
        all_weights = self.get_weight(conditional_missing_probability)
        cal_scores = self.base_y_calibration.compute_scores(x_cal, y_cal, cal_prediction).detach()
        device = x_cal.device
        beta = self.beta
        n = len(x_cal)
        quantile_level = 1. - self.alpha + beta * (n / (n+1)) + 1 / (n+1)
        quantile_level = max(0., min(quantile_level, 1.))
        quantiles = torch.Tensor([quantile_level]).to(device)
        max_score = cal_scores[~deleted_cal].max().item()
        thresholds = []
        # for i in tqdm.tqdm(range(len(x_cal))):
        curr_idx = (~deleted_cal).clone()
        # curr_idx[i] = False
        curr_weights = all_weights[curr_idx]
        curr_scores = cal_scores[curr_idx]
        # missing_probability = conditional_missing_probability[i]
        # w_test = self.get_weight(missing_probability)
        w_test = torch.quantile(all_weights, 1- beta)
        p_i = curr_weights / (curr_weights.sum() + w_test)
        p_test = w_test / (curr_weights.sum() + w_test)
        sample_weight = torch.cat([p_i, torch.tensor([p_test.item()], device=device)])
        values = torch.cat([curr_scores, torch.tensor([max_score], device=device)])
        Q = weighted_quantile(values, quantiles, sample_weight=sample_weight, old_style=True).item()
        # thresholds += [Q]
        # thresholds = torch.Tensor(thresholds).to(device)
        self.Q = Q

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ModelPrediction, **kwargs) -> UncertaintySets:
        test_calibrated_sets = self.base_y_calibration.compute_uncertainty_set_from_prediction_and_threshold(
            test_prediction, self.Q)

        return test_calibrated_sets

    def compute_scores(self, x, y, cal_prediction: ModelPrediction):
        return self.base_y_calibration.compute_scores(x, y, cal_prediction)

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> UncertaintySets:
        raise NotImplementedError("not implemented yet")

    @property
    def name(self):
        return f"pcp_{self.base_y_calibration.name}_{self.data_mask_estimator.name}_masker"

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test,
                            test_model_prediction: ModelPrediction) -> dict:
        return {
            **self.data_mask_estimator.compute_performance(x_test, z_test, full_y_test, deleted_test),
            **self.base_y_calibration.compute_performance(x_test, y, z_test, full_y_test, deleted_test,
                                                          test_model_prediction),
            'pcp_q': self.Q
        }

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal,
                                                             cal_predictions: List[ModelPrediction],
                                                             x_test,
                                                             test_prediction: List[ModelPrediction],
                                                             **kwargs) -> UncertaintySets:
        missing_probabilities = self.data_mask_estimator.predict(x_cal, z_cal)
        marginal_missing_probability = missing_probabilities.mean()
        cal_weights = (1 - marginal_missing_probability) / (1 - missing_probabilities)[~deleted_cal]
        new_deleted_cal = torch.zeros(len(cal_weights)).to(cal_weights.device)
        worst_missing_probability = torch.quantile(missing_probabilities, q=1-self.beta).item()
        test_missing_probabilities = torch.ones(len(x_test)).to(x_test.device) * worst_missing_probability
        test_weights = (1 - marginal_missing_probability) / (1 - test_missing_probabilities)
        cal_predictions = [cal_predictions[i] for i in (~deleted_cal).nonzero()]
        test_prediction = [test_prediction[i] for i in (~deleted_cal).nonzero()]
        n = len(x_cal)
        return self.base_y_calibration.jackknife_plus_construct_uncertainty_set_from_scores(x_cal[~deleted_cal],
                                                                                            y_cal[~deleted_cal],
                                                                                            z_cal[~deleted_cal],
                                                                                            new_deleted_cal,
                                                                                            cal_predictions,
                                                                                            x_test,
                                                                                            test_prediction,
                                                                                            cal_weights=cal_weights,
                                                                                            test_weights=test_weights,
                                                                                            alpha= self.alpha - self.beta * (n / (n+1)) - 1 / (n+1))

