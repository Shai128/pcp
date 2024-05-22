import abc
import os
from typing import Optional

import pandas as pd

from calibration_schemes.AbstractCalibration import Calibration
from data_utils.datasets.dataset import Dataset
from models.abstract_models.AbstractModel import Model
from models.model_utils import UncertaintySets, ModelPrediction
from utils import create_folder_if_it_doesnt_exist


def get_results_save_dir(base_results_save_dir: str, dataset_name: str, model_name: str, calibration_name: str) -> str:
    method_name = f"{model_name}_{calibration_name}".replace(" ", "_")
    return os.path.join(base_results_save_dir, dataset_name, method_name)


class ResultsHelper:
    def __init__(self, base_results_save_dir, seed):
        self.base_results_save_dir = base_results_save_dir
        self.seed = seed

    def save_performance_metrics(self,
                                 train_model_prediction: Optional[ModelPrediction],
                                 train_calibrated_uncertainty_sets: Optional[UncertaintySets],
                                 test_model_prediction: ModelPrediction,
                                 test_calibrated_uncertainty_sets: UncertaintySets,
                                 dataset: Dataset,
                                 model: Model,
                                 calibration_scheme: Calibration):
        save_dir = get_results_save_dir(self.base_results_save_dir, dataset.dataset_name, model.name,
                                        calibration_scheme.name)
        create_folder_if_it_doesnt_exist(save_dir)
        save_path = os.path.join(save_dir, f"seed={self.seed}.csv")

        results = self.compute_performance_metrics(
            train_model_prediction,
            train_calibrated_uncertainty_sets,
            test_model_prediction,
            test_calibrated_uncertainty_sets, dataset, calibration_scheme, model)
        pd.DataFrame(results, index=[self.seed]).to_csv(save_path)

    def compute_performance_metrics(self,
                                    train_model_prediction: Optional[ModelPrediction],
                                    train_calibrated_uncertainty_sets: Optional[UncertaintySets],
                                    test_model_prediction: ModelPrediction,
                                    test_calibrated_uncertainty_sets: UncertaintySets,
                                    dataset: Dataset,
                                    calibration_scheme: Calibration,
                                    model: Model) -> dict:
        if train_model_prediction is not None and train_calibrated_uncertainty_sets is not None:
            train_results = self.compute_performance_metrics_on_data(dataset.x_train,
                                                                     dataset.y_train,
                                                                     dataset.z_train,
                                                                     dataset.full_y_train,
                                                                     dataset.deleted_train,
                                                                     train_model_prediction,
                                                                     train_calibrated_uncertainty_sets,
                                                                     calibration_scheme,
                                                                     model)
        else:
            train_results = {}
        test_results = self.compute_performance_metrics_on_data(dataset.x_test,
                                                                dataset.y_test,
                                                                dataset.z_test,
                                                                dataset.full_y_test,
                                                                dataset.deleted_test,
                                                                test_model_prediction,
                                                                test_calibrated_uncertainty_sets,
                                                                calibration_scheme,
                                                                model)

        train_results = {f"train {key}": train_results[key] for key in train_results}
        return {**test_results, **train_results}

    def compute_performance_metrics_on_data(self, x, y, z, full_y, deleted, model_prediction: ModelPrediction,
                                            calibrated_uncertainty_sets: UncertaintySets,
                                            calibration_scheme: Calibration, model: Model) -> dict:
        calibration_results = calibration_scheme.compute_performance(x, y, z, full_y, deleted, model_prediction)
        model_results = model.compute_performance(x, z, full_y, deleted)
        results = self.compute_performance_metrics_on_data_aux(full_y, y, deleted, model_prediction,
                                                               calibrated_uncertainty_sets)
        if 'full y2 coverage' in results and results['full y2 coverage'] < 0.55:
            print(f"warning: coverage is less than 0.55 for calibration: {calibration_scheme.name}")
        return {**calibration_results, **results, **model_results}

    @abc.abstractmethod
    def compute_performance_metrics_on_data_aux(self, full_y, y, deleted,
                                                model_prediction: ModelPrediction,
                                                calibrated_uncertainty_sets: UncertaintySets) -> dict:
        return {}
