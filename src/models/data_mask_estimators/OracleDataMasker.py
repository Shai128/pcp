import abc

import torch

from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_scaler import DataScaler
from models.ClassificationModel import ClassProbabilities
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator


class OracleDataMasker(DataMaskEstimator):

    def __init__(self, data_scaler: DataScaler, data_masker: DataCorruptionMasker, dataset_name: str, x_dim: int, z_dim: int):
        super().__init__( dataset_name, x_dim,  z_dim)
        self.data_masker = data_masker
        self.data_scaler = data_scaler

    def forward(self, x, z) -> ClassProbabilities:
        unscaled_x = self.data_scaler.unscale_x(x)
        if z is None:
            unscaled_z = None
        else:
            unscaled_z = self.data_scaler.unscale_z(z)
        corruption_probabilities = self.data_masker.get_corruption_probabilities(unscaled_x, unscaled_z)
        class_probabilities = torch.zeros(len(x), 2).to(x.device)
        class_probabilities[:, 1] = corruption_probabilities
        class_probabilities[:, 0] = 1-corruption_probabilities
        return ClassProbabilities(class_probabilities)

    @property
    def name(self) -> str:
        return "oracle"
