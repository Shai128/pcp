import abc

import torch


class CovariatesDimensionReducer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass


class ZMeanReducer(CovariatesDimensionReducer):

    def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(z.shape) == 2:
            return z.mean(dim=-1)
        else:
            return z


class MeanReducer(CovariatesDimensionReducer):

    def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        return torch.cat([x, z], dim=-1).mean(dim=-1)


class WeightedZReducer(CovariatesDimensionReducer):

    def __init__(self, z_beta_vec: torch.Tensor):
        super().__init__()
        self.z_beta_vec = z_beta_vec

    def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(z.shape) == 2:
            return z @ self.z_beta_vec.to(z.device)
        else:
            return z
