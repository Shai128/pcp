from typing import Union

import numpy as np
import torch

from data_utils.data_corruption.corruption_type import CorruptionType
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_scaler import DataScaler
from data_utils.datasets.dataset import Dataset
from data_utils.datasets.regression_dataset import RegressionDataset
from models.LinearModel import LogisticLinearModel


class ClassificationDataset(Dataset):

    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 z: torch.Tensor,
                 deleted: torch.Tensor, data_masker: DataCorruptionMasker, dataset_name: str, training_ratio: float,
                 validation_ratio: float, calibration_ratio: float, device,
                 saved_models_path: str, figures_dir: str, seed: int,
                 full_y=None):
        super().__init__(x.shape[0], data_masker, dataset_name, training_ratio,
                         validation_ratio, calibration_ratio, device, seed)
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        apply_corruption = True
        if full_y is not None:
            impute_y = False
            apply_corruption = False
            if len(full_y.shape) == 1:
                full_y = full_y.unsqueeze(-1)
            self.full_y = full_y.clone().to(device)
        else:
            impute_y = True
            self.full_y = y.clone().to(device)
        self.n_classes = self.full_y.max().int().item() + 1
        self.full_x = x.clone().to(device)
        if apply_corruption:
            x, y, z, deleted = self.apply_corruption(x, y, z, deleted)
        self.unscaled_x = x
        self.unscaled_y = y
        self.unscaled_z = z
        self._scaler = DataScaler()
        self._scaler.initialize_scalers(self.unscaled_x[self.train_idx], None,
                                        self.unscaled_z[self.train_idx])
        self._d = deleted.to(device)
        self._x = self._scaler.scale_x(self.unscaled_x).to(device)
        self._z = self._scaler.scale_z(self.unscaled_z).to(device)
        self._y = self.unscaled_y.clone().to(device)
        self.full_x = self._scaler.scale_x(self.full_x).to(device)

        if self.corruption_type in [CorruptionType.MISSING_X]:
            self._x, self._y, self._z, self._d = RegressionDataset.impute_missing_variables(self.corruption_type,
                                                                                            self._x,
                                                                                            self._y, self._z,
                                                                                            self._d,
                                                                                            self.train_idx,
                                                                                            self.validation_idx,
                                                                                            dataset_name,
                                                                                            saved_models_path,
                                                                                            figures_dir,
                                                                                            seed)
        elif self.corruption_type in [CorruptionType.MISSING_Y] and impute_y:
            self._x, self._y, self._z, self._d = ClassificationDataset.impute_missing_response(
                self._x,
                self._y, self._z,
                self._d,
                self.train_idx,
                self.validation_idx,
                dataset_name,
                saved_models_path,
                figures_dir,
                seed)

        if len(self._d.shape) == 2:
            self._d = self._d.bool().any(dim=-1)

    @classmethod
    def impute_missing_response(cls, x, y, z, d, train_idx, validation_idx, dataset_name,
                                saved_models_path, figures_dir, seed):
        device = x.device
        d_reduced = d.squeeze()
        model = LogisticLinearModel(dataset_name, saved_models_path, figures_dir, seed)
        train_mask = torch.zeros(len(d), device=device).bool()
        train_mask[train_idx] = 1
        train_mask = train_mask & ~d_reduced
        val_mask = torch.zeros(len(d), device=device).bool()
        val_mask[validation_idx] = 1
        val_mask = val_mask & ~d_reduced
        device = x.device
        x = x.clone().to(device)
        y = y.clone().to(device)
        model_x = torch.cat([x, z], dim=-1)
        model_y = y
        model.fit(model_x[train_mask], model_y[train_mask], x_val=model_x[val_mask],
                  y_val=model_y[val_mask])
        pred = model.predict(model_x).squeeze()
        y[d.squeeze(), 0] = pred[d.squeeze()]

        return x, y, z, d

    def apply_noised_y_corruption(self, x, y, z, d):
        y = y.clone()
        labels, counts = torch.unique(y, return_counts=True)
        most_likely_label = labels[torch.argmax(counts)].item()
        y[d] = most_likely_label
        return x, y, z, d

    def apply_dispersive_noised_y_corruption(self, x, y, z, d):
        y = y.clone()
        n_labels = y.max().item() + 1
        rnd_labels = torch.randint(0, n_labels, size=y.shape[0]).to(y.device)
        y[d] = rnd_labels[d]
        return x, y, z, d

    @property
    def scaler(self) -> DataScaler:
        return self._scaler
