from typing import List
import torch
from torch import nn

import utils
from models.cnn_utils import train_transform, test_transform
from models.networks import CNN
from models.qr_models.QuantileRegression import QuantileRegression


class CNNQuantileRegression(QuantileRegression):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, y_dim: int, alpha: float,
                 hidden_dims: List[int] = None, dropout: float = 0.1, batch_norm: bool = False, lr: float = 1e-3,
                 wd: float = 0., device='cpu', figures_dir=None, seed=0, train_all_q=False, scaled_y_min: float = None,
                 scaled_y_max: float = None, base_model: nn.Module = None, *args, **kwargs):
        super().__init__(dataset_name, saved_models_path, x_dim, y_dim, alpha, hidden_dims, dropout, batch_norm, lr, wd,
                         device, figures_dir, seed, train_all_q, scaled_y_min, scaled_y_max, base_model, *args,
                         **kwargs)
        if train_all_q:
            raise NotImplementedError("not implemented train all q for CNN QR")
        self._network = CNN(dataset_name, saved_models_path, x_dim, 0, 2*y_dim,
                            hidden_dims, dropout, batch_norm, **kwargs).to(device)
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        self.train_transform = train_transform
        self.test_transform = test_transform

    def _network_inference_predict(self, x, **kwargs):
        return self.network(x, transform=self.test_transform, **kwargs)

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs,
                    train_transform=self.train_transform,
                    test_transform=self.test_transform,
                    )

    @property
    def name(self) -> str:
        return "cnn_qr"
