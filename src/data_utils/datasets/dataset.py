import abc

import torch

from data_utils.data_corruption.corruption_type import get_corruption_type_from_dataset_name, CorruptionType
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_scaler import DataScaler
import numpy as np

from utils import set_seeds


class Dataset(abc.ABC):
    def __init__(self, data_size: int, data_masker: DataCorruptionMasker, dataset_name: str, training_ratio: float,
                 validation_ratio: float, calibration_ratio: float, device, seed: int):
        self.data_size = data_size
        self.dataset_name = dataset_name
        self.device = device
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.calibration_ratio = calibration_ratio
        n = self.data_size
        train_size = int(n * self.training_ratio)
        validation_size = int(n * self.validation_ratio)
        calibration_ratio = int(n * self.calibration_ratio)

        set_seeds(seed)
        rnd_idx = np.random.permutation(n)
        self.train_idx = rnd_idx[:train_size]
        self.validation_idx = rnd_idx[train_size: train_size + validation_size]
        self.cal_idx = rnd_idx[train_size + validation_size: train_size + validation_size + calibration_ratio]
        self.test_idx = rnd_idx[train_size + validation_size + calibration_ratio:]

        self._x = None
        self._y = None
        self._z = None
        self._d = None
        self.full_y = None
        self.full_x = None
        self.data_masker = data_masker
        self.corruption_type = get_corruption_type_from_dataset_name(dataset_name)

    def apply_corruption(self, x, y, z, d):
        if self.corruption_type == CorruptionType.MISSING_Y:
            return self.apply_missing_y_corruption(x, y, z, d)
        elif self.corruption_type == CorruptionType.MISSING_X:
            return self.apply_missing_x_corruption(x, y, z, d)
        elif self.corruption_type == CorruptionType.NOISED_X:
            return self.apply_noised_x_corruption(x, y, z, d)
        elif self.corruption_type == CorruptionType.NOISED_Y:
            return self.apply_noised_y_corruption(x, y, z, d)
        elif self.corruption_type == CorruptionType.DISPERSIVE_NOISED_Y:
            return self.apply_dispersive_noised_y_corruption(x, y, z, d)
        else:
            raise Exception(
                f"Dataset {self.dataset_name} does not know how to handle with corruption type: {self.corruption_type}")

    def apply_missing_y_corruption(self, x, y, z, d):
        y = y.clone()
        y[d] = np.nan
        return x, y, z, d

    def apply_missing_x_corruption(self, x, y, z, d):
        x = x.clone()
        x[d] = np.nan
        return x, y, z, d

    @abc.abstractmethod
    def apply_noised_y_corruption(self, x, y, z, d):
        pass

    @abc.abstractmethod
    def apply_dispersive_noised_y_corruption(self, x, y, z, d):
        pass

    def apply_noised_x_corruption(self, x, y, z, d):
        x = x.clone()
        x_mean = x.mean(dim=0)
        repeated_x_mean = x_mean.unsqueeze(0).repeat(x.shape[0], 1)
        repeated_x_std = x.std(dim=0).unsqueeze(0).repeat(x.shape[0], 1)
        above_median = x > x.median(dim=0).values.unsqueeze(0).repeat(x.shape[0], 1)
        # ratio = 0.5 * above_mean + 0.8*(~above_mean)
        # diff = repeated_x_mean - x
        # scaled_diff = diff *
        noised_x = x + repeated_x_std * (1 * above_median + 4 * (~above_median))
        x[d] = noised_x[d]
        return x, y, z, d

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def d(self):
        return self._d

    @x.setter
    def x(self, value):
        self._x = value

    @z.setter
    def z(self, value):
        self._z = value

    @d.setter
    def d(self, value):
        self._d = value

    @property
    def x_train(self):
        return self.x[self.train_idx]

    @property
    def x_val(self):
        return self.x[self.validation_idx]

    @property
    def x_cal(self):
        return self.x[self.cal_idx]

    @property
    def x_test(self):
        return self.full_x[self.test_idx]

    @property
    def z_train(self):
        return self.z[self.train_idx]

    @property
    def z_val(self):
        return self.z[self.validation_idx]

    @property
    def z_cal(self):
        return self.z[self.cal_idx]

    @property
    def z_test(self):
        return self.z[self.test_idx]

    @property
    def y_train(self):
        return self.y[self.train_idx]

    @property
    def y_val(self):
        return self.y[self.validation_idx]

    @property
    def y_cal(self):
        return self.y[self.cal_idx]

    @property
    def y_test(self):
        return self.y[self.test_idx]

    @property
    def full_y_train(self):
        return self.full_y[self.train_idx]

    @property
    def full_y_val(self):
        return self.full_y[self.validation_idx]

    @property
    def full_y_cal(self):
        return self.full_y[self.cal_idx]

    @property
    def full_y_test(self):
        return self.full_y[self.test_idx]

    @property
    def deleted_train(self):
        return self.d[self.train_idx]

    @property
    def deleted_val(self):
        return self.d[self.validation_idx]

    @property
    def deleted_cal(self):
        return self.d[self.cal_idx]

    @property
    def deleted_test(self):
        return self.d[self.test_idx]

    @property
    def x_dim(self):
        return self.x.shape[-1]

    @property
    def z_dim(self):
        return self.z.shape[-1]

    @property
    @abc.abstractmethod
    def scaler(self) -> DataScaler:
        pass

    @property
    def y_dim(self):
        return self.y.shape[-1]
