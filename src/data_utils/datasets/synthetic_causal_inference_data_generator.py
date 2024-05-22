import abc
from abc import ABC

import numpy as np
import scipy
import torch
from torch import nn

from data_utils.data_corruption.corruption_type import CorruptionType
from data_utils.data_corruption.covariates_dimension_reducer import WeightedZReducer
from data_utils.data_corruption.data_corruption_masker import DataCorruptionIndicatorFactory, DataCorruptionMasker
from data_utils.data_type import DataType
from data_utils.datasets.synthetic_dataset_generator import SyntheticDataGenerator
from utils import set_seeds, get_seed


class CausalInferenceCorruptionMasker(DataCorruptionMasker):
    def __init__(self, propensity_score_function):
        super().__init__()
        self.propensity_score_function = propensity_score_function

    def get_corruption_probabilities(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor):
        return 1-self.propensity_score_function(unscaled_x, unscaled_z)


class CausalInferenceDataGenerator(SyntheticDataGenerator):
    """
    1: Was initially defined in: Estimation and inference of heterogeneous treatment effects
    using random forests.
    2: A variant of the example in https://arxiv.org/pdf/2006.06138.pdf (Conformal Inference
    # of Counterfactuals and Individual Treatment Effects)
    """
    def __init__(self, x_dim: int, rho: float = 0.9):
        super().__init__()
        self.x_dim = x_dim
        self.rho = rho

    def generate_data(self, data_size: int, device='cpu', seed=0):
        curr_seed = get_seed()
        set_seeds(seed)
        x = torch.randn(data_size, self.x_dim + 1)
        fac = torch.randn(data_size)
        x = x * np.sqrt(1 - self.rho) + fac[:, None] * np.sqrt(self.rho)
        x = torch.Tensor(scipy.stats.norm.cdf(x.numpy()))
        z = x[:, -1]
        x = x[:, :-1]
        y = self.get_y_given_x_z(x, z, seed=seed)
        data_masker = CausalInferenceCorruptionMasker(self.compute_propensity_score)
        deleted = data_masker.get_corruption_mask(x, z)

        set_seeds(curr_seed)
        x, y, z, deleted = x.to(device), y.to(device), z.to(device), deleted.to(device)
        assert x.shape[1] == self.x_dim
        return x, y, z, data_masker, deleted

    def compute_propensity_score(self, x, z):
        device = z.device
        return torch.Tensor(1 + scipy.stats.beta(2, 4).cdf(z.detach().cpu().numpy())).to(device).squeeze() / 4

    def generate_with_repeats(self, generator, *inputs, repeats: int = None, seed=0):
        curr_seed = get_seed()
        if repeats is None:
            repeats = 1
            squeeze = True
        else:
            squeeze = False
        repeated_inputs = []
        device = inputs[0].device
        for input in inputs:
            repeated_inputs += [input.unsqueeze(0).repeat(repeats, 1, 1).flatten(0, 1).cpu()]
        unflatten = nn.Unflatten(0, (repeats, inputs[0].shape[0]))
        set_seeds(seed)
        result = generator(*repeated_inputs, seed=seed)
        set_seeds(curr_seed)

        result = unflatten(result).to(device)
        if squeeze:
            result = result.squeeze(0)

        return result

    def generate_z_given_x(self, x: torch.Tensor, repeats: int = None, seed=0) -> torch.Tensor:
        raise NotImplementedError()

    def get_y_given_x_z(self, x: torch.Tensor, z: torch.Tensor, repeats: int = None, seed=0) -> torch.Tensor:
        return self.generate_with_repeats(self.get_y_given_x_z_core, x, z, repeats=repeats, seed=seed)

    def get_y_given_x_z_core(self, x: torch.Tensor, z: torch.Tensor, **kwargs):
        std = -torch.log(x[:, 1] + 1e-9)
        q = torch.quantile(z, q=0.2)
        std = std * 500 * (z < q).squeeze().float() + std * (z >= q).squeeze().float()
        tau = 2 / (1 + torch.exp(-12 * (x[:, 1] - 0.5))) * 2 / (1 + torch.exp(-12 * (x[:, 2] - 0.5)))
        y = tau + std * torch.randn(x.shape[0])
        return y

    def get_y_given_x(self, x: torch.Tensor, repeats: int = None, seed=0):
        return self.generate_with_repeats(self.get_y_given_x_core, x, repeats=repeats, seed=seed)

    def get_y_given_x_core(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError()


"""
        deleted.float().mean().item()
        
        import matplotlib.pyplot as plt
        plt.scatter(reduced_z.squeeze().cpu(), y.squeeze().cpu())
        plt.xlabel("z")
        plt.ylabel("y")
        plt.show()
        
        y[deleted].var()
        y[~deleted].var()
        
        plt.scatter(z[:, 0].squeeze().cpu(), y.squeeze().cpu())
        plt.xlabel("z0")
        plt.ylabel("y")
        plt.show()
        """
