import abc
from abc import ABC

import torch
from torch import nn

from data_utils.data_corruption.corruption_type import CorruptionType
from data_utils.data_corruption.covariates_dimension_reducer import WeightedZReducer
from data_utils.data_corruption.data_corruption_masker import DataCorruptionIndicatorFactory
from data_utils.data_type import DataType
from utils import set_seeds, get_seed


class SyntheticDataGenerator(ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_data(self, data_size: int, device='cpu'):
        pass

    @abc.abstractmethod
    def get_y_given_x(self, x: torch.Tensor, repeats: int = None, seed=0):
        pass

    @abc.abstractmethod
    def get_y_given_x_z(self, x: torch.Tensor, z: torch.Tensor, repeats: int = None, seed=0) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def generate_z_given_x(self, x: torch.Tensor, repeats: int = None, seed=0):
        pass


class PartiallyLinearDataGenerator(SyntheticDataGenerator):
    max_z_dim = 10

    def __init__(self, x_dim: int, z_dim: int, corruption_type: CorruptionType):
        super().__init__()
        assert 1 <= z_dim <= PartiallyLinearDataGenerator.max_z_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.alpha = 0.8
        self.beta = 0.2
        curr_seed = get_seed()
        set_seeds(0)
        beta_vec = torch.rand(x_dim)
        self.beta_vec = (beta_vec / beta_vec.norm(p=1))
        beta_vec2 = torch.rand(x_dim)
        self.beta_vec2 = (beta_vec2 / beta_vec2.norm(p=1))
        z_beta_vec = torch.rand(z_dim)
        self.z_beta_vec = (z_beta_vec / z_beta_vec.norm(p=1))
        self.corruption_type = corruption_type
        set_seeds(curr_seed)

    def generate_data(self, data_size: int, device='cpu', seed=0):
        curr_seed = get_seed()
        set_seeds(seed)
        x = self.generate_x(data_size, seed=seed)
        z = self.generate_z_given_x(x, seed=seed)
        y = self.get_y_given_x_z(x, z, seed=seed)
        # sample = self.get_y_given_x(x, repeats=500, seed=1).to(device)
        covariates_reducer = WeightedZReducer(self.z_beta_vec)
        corruption_masker = DataCorruptionIndicatorFactory.get_corruption_masker('partially_linear_synthetic', self.corruption_type, x, z, y,
                                                                           covariates_reducer=covariates_reducer)
        deleted = corruption_masker.get_corruption_mask(x, z)
        set_seeds(curr_seed)

        x, y, z, deleted = x.to(device), y.to(device), z.to(device), deleted.to(device)
        assert x.shape[1] == self.x_dim
        assert z.shape[1] == self.z_dim
        return x, y, z, corruption_masker, deleted

    def generate_x(self, data_size: int, seed=0):
        curr_seed = get_seed()
        set_seeds(seed)
        x = torch.rand(data_size, self.x_dim) * 4 + 1
        set_seeds(curr_seed)
        return x

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
        return self.generate_with_repeats(self.generate_z_given_x_core, x, repeats=repeats, seed=seed)

    def generate_z_given_x_core(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x_reduced = x[:, 0]
        z_list = []
        for _ in range(self.z_dim):
            z_elem = torch.poisson(torch.cos(torch.randn_like(x_reduced)) ** 2 + 0.1) * 2 * (
                    torch.rand_like(x_reduced) - 0.5) + 2 * (torch.randn_like(x_reduced))
            z_list += [z_elem.unsqueeze(-1)]

        z = torch.cat(z_list, dim=-1)
        return z

    def get_y_given_x_z(self, x: torch.Tensor, z: torch.Tensor, repeats: int = None, seed=0) -> torch.Tensor:
        return self.generate_with_repeats(self.get_y_given_x_z_core, x, z, repeats=repeats, seed=seed)

    def get_y_given_x_z_core(self, x: torch.Tensor, z: torch.Tensor, **kwargs):

        reduced_x = x @ self.beta_vec2.to(x.device)
        reduced_z = z @ self.z_beta_vec.to(z.device)
        uncertainty_level = 0.5 * (reduced_z < -3).float() + 1 * ((-3 <= reduced_z) & (reduced_z < 1)).float() + 4 * (
                1 <= reduced_z).float()

        uncertainty = 2 * uncertainty_level * (torch.randn_like(reduced_z))
        y = reduced_x * 0.3 + self.alpha * reduced_z + self.beta + uncertainty

        return y

    def get_y_given_x(self, x: torch.Tensor, repeats: int = None, seed=0):
        return self.generate_with_repeats(self.get_y_given_x_core, x, repeats=repeats, seed=seed)

    def get_y_given_x_core(self, x: torch.Tensor, **kwargs):
        z = self.generate_z_given_x(x, **kwargs)
        y = self.get_y_given_x_z(x, z, **kwargs)
        return y


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
