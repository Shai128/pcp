from typing import List
import torch
from torch import nn


from models.LinearModel import LinearModel
from models.regressors.MeanRegressor import MeanRegressor
from models.model_utils import construct_interval_from_pred, two_dimensional_pinball_loss


class LinearRegressor(MeanRegressor):

    def __init__(self, dataset_name, saved_models_path, seed, figures_dir=None):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.linear_model = LinearModel(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        linear_model_x_train = z_train
        linear_model_x_val = z_val
        linear_model_y_train = y_train
        linear_model_y_val = y_val
        self.linear_model.fit(linear_model_x_train, linear_model_y_train, deleted_train, linear_model_x_val,
                              linear_model_y_val, deleted_val)

    def loss(self, y, prediction, d, epoch):
        pass

    def predict(self, x, **kwargs):
        pass

    def predict_mean(self, x, z):
        return self.linear_model.predict(z)

    @property
    def name(self) -> str:
        return "linear"
