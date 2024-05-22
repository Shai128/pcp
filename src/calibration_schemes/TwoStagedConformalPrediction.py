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


class ConservativeWeightedCalibration(Calibration):

    def __init__(self, base_proxy_calibration: Calibration, base_y_calibration: Calibration, alpha: float,
                 dataset_name: str, data_scaler: DataScaler, proxy_qr_model: PredictionIntervalModel,
                 data_mask_estimator: DataMaskEstimator):
        super().__init__(alpha)
        self.Qs = None
        self.data_mask_estimator = data_mask_estimator
        self.dataset_name = dataset_name
        self.data_scaler = data_scaler
        self.base_proxy_calibration = base_proxy_calibration
        beta = alpha / 2
        self.base_calibration_alpha = 1 - alpha + beta
        self.base_proxy_calibration.alpha = 1 - beta
        self.base_y_calibration = base_y_calibration
        self.base_y_calibration.alpha = self.base_calibration_alpha
        self.proxy_qr_model = proxy_qr_model

        self.marginal_missing_probability = None
        self.weights = None
        self.y2_cal_scores = None

    def get_mask_probabilities(self, scaled_x, scaled_z):
        return self.data_mask_estimator.predict(scaled_x, scaled_z)

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20,
            **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
                    batch_size=batch_size, n_wait=batch_size, **kwargs)
        new_deleted_train = torch.zeros(len(x_train)).to(x_train.device).bool()
        new_deleted_val = torch.zeros(len(x_val)).to(x_val.device).bool()
        self.proxy_qr_model.fit(x_train, z_train, new_deleted_train, x_val, z_val, new_deleted_val, epochs, batch_size,
                                n_wait, **kwargs)
        self.base_proxy_calibration.fit(x_train, z_train, new_deleted_train, x_val, z_val, new_deleted_val, epochs,
                                        batch_size,
                                        n_wait, **kwargs)
        self.base_y_calibration.fit(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs,
                                    batch_size,
                                    n_wait, **kwargs)
        self.data_mask_estimator.fit(x_train, z_train, deleted_train, x_val, z_val, deleted_val, epochs=epochs,
                                     batch_size=batch_size, n_wait=n_wait, **kwargs)

    @staticmethod
    def get_weight(conditional_missing_probability, marginal_missing_probability):
        return (1 - marginal_missing_probability) / (1 - conditional_missing_probability)

    @staticmethod
    def compute_calibration_params(x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ModelPrediction,
                                   data_mask_estimator,
                                   proxy_qr_model,
                                   base_proxy_calibration,
                                   base_y_calibration):
        data_mask_estimator.calibrate(x_cal, z_cal, deleted_cal)
        proxy_prediction = proxy_qr_model.construct_uncalibrated_intervals(x_cal)
        assert proxy_prediction is not None
        base_proxy_calibration.calibrate(x_cal, z_cal, z_cal, torch.zeros(len(z_cal), device=x_cal.device).bool(),
                                         proxy_prediction)
        calibrated_sets = base_proxy_calibration.construct_calibrated_uncertainty_sets(x_cal, proxy_prediction)
        idx = calibrated_sets.contains(z_cal)
        conditional_missing_probability = data_mask_estimator.predict(x_cal[idx], z_cal[idx])
        marginal_missing_probability = conditional_missing_probability.mean()
        weights = \
            ConservativeWeightedCalibration.get_weight(conditional_missing_probability, marginal_missing_probability)[
                ~deleted_cal[idx]]
        cal_scores = base_y_calibration.compute_scores(x_cal, y_cal, cal_prediction)[
            idx & ~deleted_cal].detach()
        return cal_scores, weights, marginal_missing_probability

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal, cal_prediction: ModelPrediction, **kwargs):
        super().calibrate(x_cal, y_cal, z_cal, deleted_cal, cal_prediction, **kwargs)
        self.data_mask_estimator.calibrate(x_cal, z_cal, deleted_cal)

        cal_scores, weights, marginal_missing_probability = ConservativeWeightedCalibration. \
            compute_calibration_params(x_cal, y_cal, z_cal, deleted_cal, cal_prediction,
                                       self.data_mask_estimator,
                                       self.proxy_qr_model,
                                       self.base_proxy_calibration,
                                       self.base_y_calibration
                                       )

        self.marginal_missing_probability = marginal_missing_probability
        self.weights = weights
        self.y2_cal_scores = cal_scores

    def construct_calibrated_uncertainty_sets(self, x_test: torch.Tensor,
                                              test_prediction: ModelPrediction, **kwargs) -> UncertaintySets:
        proxy_prediction = self.proxy_qr_model.construct_uncalibrated_intervals(x_test)
        proxy_intervals = self.base_proxy_calibration.construct_calibrated_uncertainty_sets(x_test,
                                                                                            proxy_prediction).intervals
        worst_missing_probabilities = torch.max(self.get_mask_probabilities(x_test, proxy_intervals[..., 0]),
                                                self.get_mask_probabilities(x_test, proxy_intervals[..., 1]))

        thresholds = ConservativeWeightedCalibration.compute_thresholds(
            x_test,
            worst_missing_probabilities,
            self.base_calibration_alpha,
            self.y2_cal_scores,
            self.weights,
            self.marginal_missing_probability,
            **kwargs
        )
        test_calibrated_sets = self.base_y_calibration.compute_uncertainty_set_from_prediction_and_threshold(
            test_prediction, thresholds)

        return test_calibrated_sets

    @staticmethod
    def compute_thresholds(x_test: torch.Tensor,
                           worst_missing_probabilities,
                           base_calibration_alpha,
                           y2_cal_scores,
                           cal_weights,
                           marginal_missing_probability,
                           **kwargs):
        device = x_test.device
        thresholds = []
        quantiles = torch.Tensor([1 - base_calibration_alpha]).to(x_test.device)
        max_score = y2_cal_scores.max().item()

        def get_weight(p):
            return (1 - marginal_missing_probability) / (1 - p)

        for i in tqdm.tqdm(range(len(x_test))):
            missing_probability = worst_missing_probabilities[i]
            w_test = get_weight(missing_probability)
            p_i = cal_weights / (cal_weights.sum() + w_test)
            p_test = w_test / (cal_weights.sum() + w_test)
            sample_weight = torch.cat([p_i, torch.tensor([p_test.item()], device=device)])
            values = torch.cat([y2_cal_scores, torch.tensor([max_score], device=device)])
            Q = weighted_quantile(values, quantiles, sample_weight=sample_weight, old_style=False).item()
            thresholds += [Q]
        return thresholds

    def compute_scores(self, x, y, cal_prediction: ModelPrediction):
        raise NotImplementedError("not implemented yet")

    def compute_uncertainty_set_from_prediction_and_threshold(self, test_prediction: ModelPrediction,
                                                              threshold) -> UncertaintySets:
        raise NotImplementedError("not implemented yet")

    def jackknife_plus_construct_uncertainty_set_from_scores(self, x_cal, y_cal, z_cal, deleted_cal, scores_cal, x_test,
                                                             model_predictions: List[ModelPrediction], **kwargs) -> UncertaintySets:
        raise Exception("not implemented yet")

    def compute_performance(self, x_test, y, z_test, full_y_test, deleted_test,
                            test_model_prediction: ModelPrediction) -> dict:
        return self.data_mask_estimator.compute_performance(x_test, z_test, full_y_test, deleted_test)

    @property
    def name(self):
        return f"two_staged_{self.base_y_calibration.name}_{self.data_mask_estimator.name}_masker"
