import torch

from data_utils.data_corruption.corruption_type import CorruptionType
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_scaler import DataScaler
from data_utils.datasets.dataset import Dataset
from models.LinearModel import LinearModel


class RegressionDataset(Dataset):

    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 z: torch.Tensor,
                 deleted: torch.Tensor, data_masker: DataCorruptionMasker, dataset_name: str, training_ratio: float,
                 validation_ratio: float, calibration_ratio: float, device,
                 saved_models_path: str, figures_dir: str, seed: int):
        super().__init__(x.shape[0], data_masker, dataset_name, training_ratio, validation_ratio, calibration_ratio,
                         device, seed)
        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        if len(z.shape) == 1:
            z = z.unsqueeze(-1)
        self.full_y = y.clone().to(device)
        self.full_x = x.clone().to(device)
        x, y, z, deleted = self.apply_corruption(x, y, z, deleted)
        self.unscaled_x = x
        self.unscaled_y = y
        self.unscaled_z = z
        self._scaler = DataScaler()
        self._scaler.initialize_scalers(self.unscaled_x[self.train_idx], self.unscaled_y[self.train_idx],
                                        self.unscaled_z[self.train_idx])
        self._d = deleted.to(device)
        self._x = self._scaler.scale_x(self.unscaled_x).to(device)
        self._y = self._scaler.scale_y(self.unscaled_y).to(device)
        self._z = self._scaler.scale_z(self.unscaled_z).to(device)
        self.full_y = self._scaler.scale_y(self.full_y).to(device)
        self.full_x = self._scaler.scale_x(self.full_x).to(device)

        if self.corruption_type in [CorruptionType.MISSING_Y, CorruptionType.MISSING_X]:
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

        if len(self._d.shape) == 2:
            self._d = self._d.bool().any(dim=-1)

    def apply_noised_y_corruption(self, x, y, z, d):
        y = y.clone()
        y_mean = y.mean()
        noised_y = (y + y_mean) / 2
        y[d] = noised_y[d]
        return x, y, z, d

    def apply_dispersive_noised_y_corruption(self, x, y, z, d):
        y = y.clone()
        noise = 5*y.std() * torch.randn_like(y)
        noised_y = y + noise
        y[d] = noised_y[d]
        return x, y, z, d

    @staticmethod
    def impute_missing_variables(corruption_type, x, y, z, d, train_idx, validation_idx, dataset_name,
                                 saved_models_path, figures_dir, seed):
        x = x.clone()
        y = y.clone()
        device = x.device
        if corruption_type == CorruptionType.MISSING_Y:
            model_x = torch.cat([x, z], dim=-1)
            model_y = y
        elif corruption_type == CorruptionType.MISSING_X:
            always_existing_features = ~(d.any(dim=0))
            model_x = torch.cat([x[:, always_existing_features], y, z], dim=-1)
            model_y = x
        else:
            raise Exception(f"don't know how to impute missing data with corruption type: {corruption_type}")
        estimated = RegressionDataset.impute_missing_variable(model_x, model_y, d, train_idx, validation_idx,
                                                              dataset_name,
                                                              saved_models_path, figures_dir, seed, device)
        model_y[d] = estimated[d]
        return x, y, z, d

    @staticmethod
    def impute_missing_variable(observed_variables, missing_variables, d, train_idx, validation_idx, dataset_name,
                                saved_models_path, figures_dir, seed, device):
        if len(d.shape) == 2:
            d_reduced = d.bool().any(dim=-1)
        else:
            d_reduced = d
        train_mask = torch.zeros(len(d), device=device).bool()
        train_mask[train_idx] = 1
        train_mask = train_mask & ~d_reduced
        val_mask = torch.zeros(len(d), device=device).bool()
        val_mask[validation_idx] = 1
        val_mask = val_mask & ~d_reduced
        imputation_model = LinearModel(dataset_name, saved_models_path, figures_dir, seed)
        model_x, model_y = observed_variables, missing_variables
        imputation_model.fit(model_x[train_mask], model_y[train_mask], None, model_x[val_mask], model_y[val_mask], None)
        estimated = imputation_model.predict(model_x)
        missing_variables = missing_variables.clone()
        missing_variables[d] = estimated[d]
        return missing_variables

    @property
    def scaler(self) -> DataScaler:
        return self._scaler

    @property
    def scaled_y_min(self) -> float:
        return self.full_y.min().item()

    @property
    def scaled_y_max(self) -> float:
        return self.full_y.max().item()
