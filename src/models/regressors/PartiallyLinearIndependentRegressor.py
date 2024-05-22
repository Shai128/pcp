from typing import List

import numpy as np
import torch
from torch import nn

from models.LinearModel import LinearModel
from models.regressors.MeanRegressor import MeanRegressor
from models.networks import BaseModel
from utils import corr, HSIC, filter_missing_values, MMD


class PartiallyLinearIndependentRegressor(MeanRegressor):

    def __init__(self, dataset_name, saved_models_path, x_dim, z_dim, hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False, lr: float = 1e-3, wd: float = 0., device='cpu',
                 figures_dir=None, seed=0):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        self._network = BaseModel(x_dim, 1, hidden_dims=hidden_dims, dropout=dropout,
                                  batch_norm=batch_norm).to(device)
        self.coeffs = nn.Parameter(torch.tensor([0.]*z_dim, requires_grad=True, device=device))
        params = self.parameters()
        self._optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        self.lr = lr
        self.wd = wd
        self.linear_model = LinearModel(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.z_dim = z_dim

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20,
            train_uncalibrated_intervals=None,
            val_uncalibrated_intervals=None, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        epochs = 5000
        assert train_uncalibrated_intervals is not None
        assert val_uncalibrated_intervals is not None
        batch_size = 1024
        # x_train, y_train, deleted_train = filter_missing_values(x_train, y_train, deleted_train)
        # x_val, y_val, deleted_val = filter_missing_values(x_val, y_val, deleted_val)
        y_train = torch.cat([y_train, train_uncalibrated_intervals.intervals], dim=-1)
        y_val = torch.cat([y_val, val_uncalibrated_intervals.intervals], dim=-1)
        new_x_train = torch.cat([z_train, x_train], dim=-1)
        new_x_val = torch.cat([z_val, x_val], dim=-1)
        self.fit_xy(new_x_train, y_train, deleted_train, new_x_val, y_val, deleted_val, epochs, batch_size, n_wait)
        # print(f"alpha: {self.alpha}")
        # linear_model_x_train = torch.stack([self.network(x_train).squeeze(), y_train[:, 0]]).T
        # linear_model_x_val = torch.stack([self.network(x_val).squeeze(), y_val[:, 0]]).T
        # linear_model_y_train = y_train[:, 1]
        # linear_model_y_val = y_val[:, 1]
        # self.linear_model.fit(linear_model_x_train, linear_model_y_train, deleted_train, linear_model_x_val,
        #                       linear_model_y_val, deleted_val, epochs, batch_size, n_wait)

    def loss(self, y, prediction, deleted, epoch):
        coeffs, z, pred = prediction
        pred = pred.squeeze()
        z = z.squeeze()
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        linear_part = z @ coeffs

        pred = pred + linear_part
        interval_min = y[:, -2]
        interval_max = y[:, -1]
        y = y[:, 0]
        error = y - pred
        mse = (error ** 2).mean()

        if epoch < 10:
            return mse

        # y2_covered = (y2 <= interval_max) & (y2 >= interval_min)
        interval_lengths = interval_max - interval_min
        y2_interval_estimate = torch.rand_like(interval_lengths) * interval_lengths + interval_min
        # y2_interval_estimate = (y[deleted, -1] + y[deleted, -2])/2
        deleted_error = y2_interval_estimate[deleted] - pred[deleted]
        imputed_error = torch.zeros_like(error)
        imputed_error[~deleted] = error[~deleted]
        imputed_error[deleted] = deleted_error
        # imputed_y2 = torch.zeros_like(y)
        # imputed_y2[~deleted] = y[~deleted]
        # imputed_y2[deleted] = y2_interval_estimate[deleted]
        # cover_dep_penalty = corr(y2_covered.float(), imputed_error).abs()
        # y2_dep_penalty = corr(imputed_y2, imputed_error).abs() + HSIC(imputed_y2, imputed_error).sqrt()
        y2_dep_penalty = corr(y[~deleted], error[~deleted]).abs() + HSIC(y[~deleted], error[~deleted]).sqrt()
        y2_dep_penalty += corr(y2_interval_estimate[deleted], deleted_error).abs() + HSIC(
            y2_interval_estimate[deleted], deleted_error).sqrt()
        mask_dep_penalty = corr(deleted.float(), imputed_error).abs() + HSIC(deleted.float(),
                                                                             imputed_error).sqrt()  # MMD(error[deleted], error[~deleted])  #

        loss = mse + y2_dep_penalty + mask_dep_penalty
        return loss

    def predict(self, x, **kwargs):
        z, x = x[:, :self.z_dim], x[:, self.z_dim:]
        return self.coeffs, z, self.network(x).squeeze()

    def predict_mean(self, x, z):
        self.eval()
        z = z.squeeze()
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        linear_part = z @ self.coeffs
        model_output = self.network(x).squeeze()
        return model_output + linear_part
        # linear_model_input = torch.stack([model_output, y[:, 0]]).T
        # return self.linear_model.predict(linear_model_input)

    @property
    def name(self) -> str:
        return "indep_partially_linear"
