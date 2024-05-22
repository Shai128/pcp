from typing import List
import torch
from torch import nn

from models.LinearModel import LinearModel
from models.regressors.MeanRegressor import MeanRegressor
from models.networks import BaseModel


class PartiallyLinearRegressor(MeanRegressor):

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

    def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
        super().fit(x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=epochs,
            batch_size=batch_size, n_wait=n_wait, **kwargs)
        new_x_train = torch.cat([z_train, x_train], dim=-1)
        new_x_val = torch.cat([z_val, x_val], dim=-1)
        self.fit_xy(new_x_train, y_train, deleted_train, new_x_val, y_val, deleted_val, epochs, batch_size, n_wait)
        linear_model_x_train = torch.cat([self.network(x_train).reshape(len(x_train), 1), z_train], dim=-1)
        linear_model_x_val = torch.cat([self.network(x_val).reshape(len(x_val), 1), z_val], dim=-1)
        linear_model_y_train = y_train
        linear_model_y_val = y_val
        self.linear_model.fit(linear_model_x_train, linear_model_y_train, deleted_train, linear_model_x_val, linear_model_y_val, deleted_val)

    def loss(self, y, prediction, d, epoch):
        coeffs, z, pred = prediction
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        linear_part = z @ coeffs
        gt = (y.squeeze() - linear_part.squeeze()).squeeze()
        return ((gt - pred) ** 2).mean()

    def predict(self, x, **kwargs):
        z, x = x[:, :self.z_dim], x[:, self.z_dim:]
        return self.coeffs, z, self.network(x).squeeze()

    def predict_mean(self, x, z):
        self.eval()
        model_output = self.network(x)
        linear_model_input = torch.cat([model_output.reshape(len(x), 1), z], dim=-1)
        return self.linear_model.predict(linear_model_input)

    @property
    def name(self) -> str:
        return "partially_linear"
