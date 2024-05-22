from typing import List
import torch
from torch import nn

import utils
from models.abstract_models.NetworkLearningModel import NetworkLearningModel
from models.qr_models.PredictionIntervalModel import PredictionIntervalModel, PredictionIntervals
from models.model_utils import construct_interval_from_pred, two_dimensional_pinball_loss, batch_pinball_loss
from models.networks import BaseModel


class QuantileRegression(NetworkLearningModel, PredictionIntervalModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, y_dim: int, alpha: float,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False,
                 lr: float = 1e-3, wd: float = 0., device='cpu', figures_dir=None,
                 seed=0, train_all_q=False,
                 scaled_y_min: float = None, scaled_y_max: float = None,
                 base_model: nn.Module = None,
                 *args, **kwargs
                 ):
        if y_dim == 0:
            raise Exception("QuantileRegression got y_dim=0")
        NetworkLearningModel.__init__(self, dataset_name, saved_models_path, figures_dir, seed)
        alpha = 1 - ((1 - alpha) ** (1 / y_dim))
        PredictionIntervalModel.__init__(self, alpha)

        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        if train_all_q:
            out_dim = y_dim
            in_dim = x_dim + 1
        else:
            out_dim = 2 * y_dim
            in_dim = x_dim
        if base_model is None:
            self._network = BaseModel(in_dim, out_dim, hidden_dims=hidden_dims, dropout=dropout,
                                      batch_norm=batch_norm).to(device)
        else:
            self._network = base_model
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        self.lr = lr
        self.wd = wd
        self.train_all_q = train_all_q
        self.device = device
        self.scaled_y_min = scaled_y_min
        self.scaled_y_max = scaled_y_max

    def get_learned_quantile_levels(self):
        if self.train_all_q:
            return torch.arange(0.02, 0.99, 0.005, device=self.device)
        else:
            return torch.Tensor([self.alpha / 2, 1 - self.alpha / 2]).to(self.device)

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)

    def _network_inference_predict(self, x, **kwargs):
        return self.network(x, **kwargs)

    def _network_train_predict(self, x, **kwargs):
        return self.network(x, **kwargs)

    def __estimate_quantiles(self, x, quantile_levels, is_train, **kwargs):
        assert self.train_all_q
        quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(x.shape[0], 1).flatten(0, 1)
        x_rep = x.unsqueeze(1).repeat(1, quantile_levels.shape[0], 1).flatten(0, 1)
        unflatten = torch.nn.Unflatten(dim=0, unflattened_size=(x.shape[0], quantile_levels.shape[0]))
        network_input = torch.cat([x_rep, quantile_levels_rep.unsqueeze(-1)], dim=-1)
        if is_train:
            model_output = self._network_train_predict(network_input, **kwargs)
        else:
            model_output = self._network_inference_predict(network_input, **kwargs)

        quantiles = unflatten(model_output).squeeze(-1)
        return quantiles

    def get_quantile_function(self, x, extrapolate_quantiles=False):
        assert self.train_all_q
        quantile_levels, _ = self.get_learned_quantile_levels().sort()
        quantiles = self.__estimate_quantiles(x, quantile_levels, is_train=False)
        quantile_levels = quantile_levels.detach().squeeze()
        quantiles = quantiles.detach()
        quantile_functions = utils.batch_estim_dist(quantiles, quantile_levels, self.scaled_y_min,
                                                    self.scaled_y_max,
                                                    smooth_tails=True, tau=0.01,
                                                    extrapolate_quantiles=extrapolate_quantiles)
        return quantile_functions

    def construct_uncalibrated_intervals(self, x: torch.Tensor, **kwargs) -> PredictionIntervals:
        with torch.no_grad():
            if self.train_all_q:
                alpha_rep = torch.ones((x.shape[0], 1), device=x.device) * self.alpha
                _, inverse_cdf = self.get_quantile_function(x)
                q_low = inverse_cdf(alpha_rep / 2)
                q_high = inverse_cdf(1 - alpha_rep / 2)
                intervals = torch.stack([q_low, q_high]).T
                return PredictionIntervals(intervals.squeeze())
            else:
                pred = self._network_inference_predict(x, **kwargs).detach()
                return PredictionIntervals(construct_interval_from_pred(pred).squeeze())

    def loss(self, y, prediction, d, epoch):
        if self.train_all_q:
            quantile_levels = self.get_learned_quantile_levels()
            quantiles = prediction
            quantile_levels_rep = quantile_levels.unsqueeze(0).repeat(y.shape[0], 1).flatten(0, 1)
            y_rep = y.squeeze().unsqueeze(1).repeat(1, quantile_levels.shape[0]).flatten(0, 1)
            return batch_pinball_loss(quantile_levels_rep, quantiles.flatten(0, 1), y_rep)
        else:
            return two_dimensional_pinball_loss(y, prediction, self.alpha)

    def predict(self, x, **kwargs):
        if self.train_all_q:
            quantile_levels = self.get_learned_quantile_levels()
            return self.__estimate_quantiles(x, quantile_levels, is_train=True, **kwargs)
        else:
            return self._network_train_predict(x, **kwargs)

    @property
    def name(self) -> str:
        return "qr"
