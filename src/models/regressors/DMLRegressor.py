from typing import List
import torch
from torch import nn

from models.regressors.MeanRegressor import MeanRegressor
from models.networks import BaseModel


class DMLRegressor(MeanRegressor):

    def __init__(self, dataset_name, saved_models_path, x_dim, y_dim, hidden_dims: List[int] = None,
                 dropout: float = 0.1,
                 batch_norm: bool = False, lr: float = 1e-3, wd: float = 0., device='cpu',
                 figures_dir=None, seed=0):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        self._network = BaseModel(x_dim, 2, hidden_dims=hidden_dims, dropout=dropout,
                                  batch_norm=batch_norm).to(device)
        self._optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        self.lr = lr
        self.wd = wd
        self.theta_hat = None

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        assert z_val.shape[1] == 1
        new_y_train = torch.cat([y_train, z_train], dim=-1)
        new_y_val = torch.cat([y_val, z_val], dim=-1)
        self.fit_xy(x_train, new_y_train, deleted_train, x_val, new_y_val, deleted_val, epochs, batch_size, n_wait)

    def loss(self, y, prediction, d, epoch):
        prediction = prediction.squeeze()
        return ((y.squeeze() - prediction) ** 2).mean()

    def predict(self, x, **kwargs):
        return self.network(x).squeeze()

    def predict_mean(self, x, z):
        self.eval()
        d = z.squeeze()
        model_output = self.network(x).squeeze()
        m_x = model_output[:, 0]
        l_x = model_output[:, 1]
        g_x = l_x - self.theta_hat * m_x
        return g_x + d * self.theta_hat

    def calibrate(self, x_cal, y_cal, z_cal, deleted_cal):
        model_output = self.network(x_cal).squeeze()
        m_x = model_output[:, 0]
        l_x = model_output[:, 1]
        d = z_cal

        v_hat = d - m_x
        u_hat = y_cal - l_x
        self.theta_hat = (v_hat * u_hat).mean() / (v_hat * v_hat).mean()

    @property
    def name(self) -> str:
        return "dml"
