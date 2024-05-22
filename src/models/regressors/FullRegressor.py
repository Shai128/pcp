from typing import List
import torch

from models.regressors.MeanRegressor import MeanRegressor
from models.networks import BaseModel


class FullRegressor(MeanRegressor):
    def __init__(self, dataset_name, saved_models_path, x_dim, z_dim, hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False, lr: float = 1e-3, wd: float = 0., device = 'cpu',
                 figures_dir=None, seed=0):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        self._network = BaseModel(x_dim+z_dim, 1, hidden_dims=hidden_dims, dropout=dropout,
                               batch_norm=batch_norm).to(device)
        params = self.parameters()
        self._optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
        self.lr = lr
        self.wd = wd
        self.z_dim = 0

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        if z_train is not None and z_val is not None:
            assert self.z_dim != 0
            x_train = torch.cat([x_train, z_train], dim=-1)
            x_val = torch.cat([x_val, z_val], dim=-1)
        else:
            assert self.z_dim == 0
        super().fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs, batch_size, n_wait)

    def loss(self, y, prediction, d, epoch):
        pred = prediction.squeeze()
        return ((y.squeeze() - pred) ** 2).mean()

    def predict(self, x, **kwargs):
        return self.network(x).squeeze()

    def predict_mean(self, x, z):
        if z is not None:
            x = torch.cat([x, z], dim=-1)
        return self.network(x).squeeze()

    @property
    def name(self) -> str:
        return "full"

