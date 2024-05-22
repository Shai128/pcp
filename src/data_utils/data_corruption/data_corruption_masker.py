# global max_val1, max_val2, threshold
import abc
import dataclasses
from abc import ABC

import numpy as np
import torch

from data_utils.data_corruption.corruption_type import CorruptionType
from data_utils.data_corruption.covariates_dimension_reducer import CovariatesDimensionReducer, ZMeanReducer
from data_utils.data_type import DataType
from utils import get_seed, set_seeds


@dataclasses.dataclass
class DataCorruptionInfo:
    def __init__(self, max_val1: float, max_val2: float, threshold: float, power: float, extreme_max_val: float,
                 min_val: float):
        self.max_val1 = max_val1
        self.max_val2 = max_val2
        self.threshold = threshold
        self.power = power
        self.extreme_max_val = extreme_max_val
        self.min_val = min_val


class DataCorruptionMasker(ABC):

    @abc.abstractmethod
    def get_corruption_probabilities(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor):
        pass

    def get_corruption_mask(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor,
                            seed: int = 0) -> torch.Tensor:
        probability_to_delete = self.get_corruption_probabilities(unscaled_x, unscaled_z).squeeze()
        if seed is not None:
            curr_seed = get_seed()
            set_seeds(seed)
        mask = torch.rand_like(probability_to_delete) < probability_to_delete
        if seed is not None:
            set_seeds(curr_seed)
        return mask


class DummyDataCorruptionMasker(DataCorruptionMasker):

    def get_corruption_probabilities(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor):
        return torch.zeros(len(unscaled_z)).to(unscaled_x.device)


class OracleDataCorruptionMasker(DataCorruptionMasker):

    def __init__(self, unscaled_x, unscaled_z, mask_probability):
        self.unscaled_x = unscaled_x
        self.unscaled_z = unscaled_z
        self.mask_probability = mask_probability
        if len(unscaled_x.shape) > 2:
            unscaled_x = torch.flatten(unscaled_x, start_dim=1)
        self.covariates = torch.cat([unscaled_x, unscaled_z], dim=-1)

    def get_corruption_probabilities(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor):
        if len(unscaled_z.shape) == 1:
            unscaled_z = unscaled_z.unsqueeze(-1)
        if len(unscaled_x.shape) > 2:
            unscaled_x = torch.flatten(unscaled_x, start_dim=1)
        test_covariates = torch.cat([unscaled_x, unscaled_z], dim=-1)
        self.covariates = self.covariates.to(test_covariates.device)
        mask_probabilities = []
        max_n_zero = 0
        min_n_zero = np.inf
        for covariate in test_covariates:
            diffs = (covariate.unsqueeze(0) - self.covariates).abs().mean(dim=-1)
            n_zero = (diffs == 0.).float().sum().item()
            max_n_zero = max(max_n_zero, n_zero)
            min_n_zero = min(min_n_zero, n_zero)
            sample_idx = torch.argmin((covariate.unsqueeze(0) - self.covariates).abs().mean(dim=-1))
            mask_probabilities += [self.mask_probability[sample_idx].item()]
        # print("max_n_zero: ", max_n_zero)
        # print("min_n_zero: ", min_n_zero)

        return torch.Tensor(mask_probabilities).to(unscaled_x.device)


class DefaultDataCorruptionMasker(DataCorruptionMasker):
    def __init__(self, dataset_name: str, covariates_dimension_reducer: CovariatesDimensionReducer,
                 unscaled_full_x: torch.Tensor,
                 unscaled_full_z: torch.Tensor,
                 marginal_masking_ratio: float = 0.2):
        self.covariates_dimension_reducer = covariates_dimension_reducer
        self.dataset_name = dataset_name
        self.unscaled_full_z = unscaled_full_z.clone()
        self.data_masking_info = self.__compute_data_corruption_info(unscaled_full_x, unscaled_full_z,
                                                                     marginal_masking_ratio)
        self.marginal_masking_ratio = marginal_masking_ratio
        # self.marginal_masking_ratio = 0.

    def get_corruption_probabilities(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor):
        unscaled_z = unscaled_z.clone()
        # if 'blog' in self.dataset_name:
        #     z_copy = self.unscaled_full_z.clone()
        #     unique_vals = unscaled_z.unique()
        #     for val in unique_vals:
        #         q_level = (z_copy <= val).float().mean().item()
        #         new_val = torch.quantile(z_copy, q=1-q_level)
        #         unscaled_z[unscaled_z == val] = new_val
        z = self.covariates_dimension_reducer(unscaled_x, unscaled_z)
        z = z - self.data_masking_info.min_val
        vals = z.clone()
        vals[vals >= self.data_masking_info.max_val2] = self.data_masking_info.max_val2
        vals /= self.data_masking_info.max_val1
        vals[z < self.data_masking_info.threshold] = 0

        probability_to_delete = vals ** self.data_masking_info.power
        z_for_extreme_vals = z / self.data_masking_info.extreme_max_val
        z_for_extreme_vals[z_for_extreme_vals <= 1] = 1
        extreme_idx = z > self.data_masking_info.extreme_max_val
        extreme_idx_new_value = 0.05 * (1 - 1 / z_for_extreme_vals) + 0.95
        probability_to_delete[extreme_idx] = extreme_idx_new_value[extreme_idx]
        return probability_to_delete

    def __compute_data_corruption_info(self, unscaled_full_x: torch.Tensor, unscaled_full_z: torch.Tensor,
                                       marginal_masking_ratio: float) -> DataCorruptionInfo:
        unscaled_full_z = unscaled_full_z.clone()
        # if 'blog' in self.dataset_name:
        #     z_copy = self.unscaled_full_z.clone()
        #     unique_vals = unscaled_full_z.unique()
        #     unscaled_full_z = unscaled_full_z.clone()
        #     for val in unique_vals:
        #         q_level = (z_copy <= val).float().mean().item()
        #         new_val = torch.quantile(z_copy, q=1-q_level)
        #         unscaled_full_z[unscaled_full_z == val] = new_val

        z = self.covariates_dimension_reducer(unscaled_full_x, unscaled_full_z)
        z_min = min(torch.quantile(z, q=0.05).item(), 0)
        if z_min < 0:
            z = z - z_min
        extreme_max_val = torch.quantile(z, q=0.999).item() * 1.2
        max_val1 = torch.quantile(z, q=0.9).item()
        max_val2 = torch.quantile(z, q=0.85).item()
        if abs(max_val1 - max_val2) < 0.001:
            print(f"warning: abs(max_val1 - max_val2) is too small: {abs(max_val1 - max_val2)}")
            # print("choosing max_val2 < max_val1")
            # z_vals = torch.unique(z)
            # max_val2 = z_vals[z_vals < max_val1].max()

        threshold = torch.quantile(z, q=0.75).item()
        # import matplotlib.pyplot as plt
        # plt.hist(z)
        # plt.show()
        vals = z.clone()
        vals[z >= max_val2] = max_val2
        vals /= max_val1
        vals[z < threshold] = 0
        vals[vals < 0] = 0
        # vals = vals[vals > 0]

        powers = torch.cat([torch.arange(0.01, 10, 0.01)]).to(vals.device)
        vals_rep = vals.unsqueeze(1).repeat(1, len(powers))
        masking_ratio_deviation_rep = (torch.pow(vals_rep, powers).mean(dim=0) - marginal_masking_ratio).abs()
        power_idx = masking_ratio_deviation_rep.argmin().item()
        power = powers[power_idx].item()

        masking_ratio_deviation = masking_ratio_deviation_rep[power_idx].item()
        if masking_ratio_deviation > 0.001:
            print(f"warning: the masking ratio deviation is too large: {np.round(masking_ratio_deviation, 4)}")

        data_masking_info = DataCorruptionInfo(max_val1, max_val2, threshold, power, extreme_max_val, z_min)
        return data_masking_info


class ResponseDataCorruptionMasker(DefaultDataCorruptionMasker):
    pass


class CovariateDataCorruptionMask(DefaultDataCorruptionMasker):
    def __init__(self, dataset_name:str , covariates_dimension_reducer: CovariatesDimensionReducer, unscaled_full_x: torch.Tensor,
                 unscaled_full_z: torch.Tensor,
                 unscaled_full_y: torch.Tensor):
        super().__init__(dataset_name, covariates_dimension_reducer, unscaled_full_x, unscaled_full_z)
        if len(unscaled_full_y.shape) == 2:
            unscaled_full_y = unscaled_full_y.mean(dim=-1)
        if unscaled_full_x.shape[1] < 5:
            raise Exception("cannot apply corruptions to feature vectors with dimension less than 5")
        x_y_correlation = [abs(np.corrcoef(unscaled_full_x[:, i], unscaled_full_y)[0, 1]) for i in
                           range(unscaled_full_x.shape[1])]
        for i in range(len(x_y_correlation)):
            if np.isnan(x_y_correlation[i]):
                x_y_correlation[i] = 0
        n_features_to_mask = int(0.2 * len(x_y_correlation))
        self.highest_correlation_x_idx = np.argsort(x_y_correlation)[::-1][:n_features_to_mask]
        assert x_y_correlation[self.highest_correlation_x_idx[0]] == np.max(x_y_correlation)

    def get_corruption_mask(self, unscaled_x: torch.Tensor, unscaled_z: torch.Tensor, seed: int = 0):
        one_dimensional_mask = super().get_corruption_mask(unscaled_x, unscaled_z, seed).squeeze()
        # mask = torch.zeros_like(unscaled_x).bool()
        sample_idx_mask = one_dimensional_mask.unsqueeze(1).repeat(1, unscaled_x.shape[1])
        feature_idx_mask = torch.zeros(unscaled_x.shape[1], dtype=torch.bool).to(unscaled_x.device)
        feature_idx_mask[self.highest_correlation_x_idx.tolist()] = True
        feature_idx_mask = feature_idx_mask.unsqueeze(0).repeat(unscaled_x.shape[0], 1)
        mask = feature_idx_mask & sample_idx_mask
        return mask


class DataCorruptionIndicatorFactory:

    @staticmethod
    def get_corruption_masker(dataset_name: str, corruption_type: CorruptionType,
                              unscaled_full_x: torch.Tensor, unscaled_full_z: torch.Tensor,
                              unscaled_full_y: torch.Tensor,
                              covariates_reducer: CovariatesDimensionReducer = None) -> DataCorruptionMasker:
        if covariates_reducer is None:
            covariates_reducer = ZMeanReducer()

        if corruption_type == CorruptionType.MISSING_X or corruption_type == CorruptionType.NOISED_X:
            data_masker = CovariateDataCorruptionMask(dataset_name, covariates_reducer, unscaled_full_x, unscaled_full_z,
                                                      unscaled_full_y)
        elif corruption_type == CorruptionType.MISSING_Y or corruption_type == CorruptionType.NOISED_Y or corruption_type == CorruptionType.DISPERSIVE_NOISED_Y:
            data_masker = ResponseDataCorruptionMasker(dataset_name, covariates_reducer, unscaled_full_x,
                                                       unscaled_full_z)
        else:
            raise Exception(f"don't know what data masker to create for corruption type: {corruption_type.name}")
        return data_masker
