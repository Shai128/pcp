import abc
from abc import ABC

from models.abstract_models.NetworkLearningModel import NetworkLearningModel


class MeanRegressor(NetworkLearningModel, ABC):

    def __init__(self, dataset_name, saved_models_path, seed, figures_dir=None):
        NetworkLearningModel.__init__(self, dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.mean_regressor_fit_count = 0

    @abc.abstractmethod
    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20, **kwargs):
        if self.mean_regressor_fit_count > 0:
            print(
                f"warning: {self.name} regressor model was fitted {self.mean_regressor_fit_count} times already and is now fitted once again.")
        self.mean_regressor_fit_count += 1

    @abc.abstractmethod
    def predict_mean(self, x, z):
        pass

    @property
    def name(self) -> str:
        return "mean_regressor"

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal):
        pass
