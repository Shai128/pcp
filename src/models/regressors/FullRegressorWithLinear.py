from typing import List
import torch

from models.LinearModel import LinearModel
from models.regressors.FullRegressor import FullRegressor
from models.regressors.MeanRegressor import MeanRegressor


class FullRegressorWithLinear(MeanRegressor):
    def __init__(self, dataset_name, saved_models_path, x_dim, z_dim, hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False, lr: float = 1e-3, wd: float = 0., device = 'cpu',
                 figures_dir=None, seed=0):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.full_regressor = FullRegressor(dataset_name, saved_models_path, x_dim, z_dim, hidden_dims, dropout,
                 batch_norm, lr, wd, device,figures_dir, seed)
        self.linear_model = LinearModel(dataset_name, saved_models_path, figures_dir, seed)

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        new_x_train = torch.cat([x_train, z_train], dim=-1)
        new_x_val = torch.cat([x_val, z_val], dim=-1)
        self.full_regressor.fit_xy(new_x_train, y_train, deleted_train, new_x_val, y_val, deleted_val, epochs, batch_size, n_wait)

        linear_model_x_train = torch.cat([self.full_regressor.predict_mean(x_train, z_train).reshape(len(x_train), 1), z_train], dim=-1)
        linear_model_x_val = torch.cat([self.full_regressor.predict_mean(x_val, z_val).reshape(len(x_val), 1), z_val], dim=-1)
        linear_model_y_train = y_train
        linear_model_y_val = y_val
        self.linear_model.fit(linear_model_x_train, linear_model_y_train, deleted_train, linear_model_x_val, linear_model_y_val, deleted_val)

    def loss(self, y, prediction, d, epoch):
        pass

    def predict(self, x, **kwargs):
        pass

    def predict_mean(self, x, z):
        self.eval()
        mean_pred = self.full_regressor.predict_mean(x, z).squeeze()
        linear_model_input = torch.cat([mean_pred.reshape(len(x), 1), z], dim=-1)
        return self.linear_model.predict(linear_model_input)

    @property
    def name(self) -> str:
        return "full_with_linear"

    @property
    def save_name(self) -> str:
        return "full"

