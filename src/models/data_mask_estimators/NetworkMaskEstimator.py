import abc
from abc import ABC
from typing import List, Optional

import torch

from models.ClassificationModel import ClassProbabilities, ClassificationModel
from models.classifiers.NetworkClassifier import NetworkClassifier
from models.classifiers.RFClassifier import RFClassifier
from models.classifiers.XGBClassifier import XGBClassifier
from models.classifiers.CNNClassifier import CNNClassifier
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator


class TabularDataMaskEstimator(DataMaskEstimator, ABC):
    def __init__(self, dataset_name: str, x_dim: int, z_dim: int):
        super().__init__(dataset_name, x_dim, z_dim)

    def get_model_x(self, x, z: Optional[torch.Tensor]):
        if z is None and self.use_z:
            raise Exception(f"{self.name} was expecting z")

        if z is None or not self.use_z:
            model_x = x
        elif x is None or not self.use_x:
            model_x = z
        else:
            if len(z.shape) == 1:
                z = z.unsqueeze(-1)
            model_x = torch.cat([z, x], dim=-1)
        return model_x

    def fit(self, x_train, z_train: Optional[torch.Tensor], deleted_train, x_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20,
            **kwargs):
        device = x_train.device
        super().fit(x_train, z_train, deleted_train, x_val, z_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)
        model_x_train = self.get_model_x(x_train, z_train)
        model_y_train = deleted_train
        model_d_train = torch.zeros(len(model_y_train), device=device)
        model_x_val = self.get_model_x(x_val, z_val)
        model_y_val = deleted_val
        model_d_val = torch.zeros(len(model_y_val), device=device)
        self.classifier.fit(model_x_train, model_y_train, model_d_train, model_x_val, model_y_val, model_d_val,
                            epochs, batch_size, n_wait)

    def forward(self, x, z) -> ClassProbabilities:
        model_x = self.get_model_x(x, z)
        prediction = self.classifier.estimate_probabilities(model_x)
        return prediction

    def compute_performance(self, x_test, z_test, full_y_test, deleted_test):
        model_x = self.get_model_x(x_test, z_test)
        model_y = deleted_test
        model_d_train = torch.zeros(len(model_x), device=x_test.device)
        classifier_performance = self.classifier.compute_performance(model_x, None, model_y, model_d_train)
        masker_performance = super().compute_performance(x_test, z_test, full_y_test, deleted_test)
        return {
            **classifier_performance,
            **masker_performance
        }

    @property
    @abc.abstractmethod
    def classifier(self) -> ClassificationModel:
        pass


class NetworkMaskEstimator(TabularDataMaskEstimator):

    def __init__(self, dataset_name, saved_models_path, x_dim: int,
                 z_dim: int,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False,
                 lr: float = 1e-3, wd: float = 0., device='cpu', figures_dir=None,
                 seed=0):
        super().__init__(dataset_name, x_dim, z_dim)
        n_classes = 2
        self._classifier = NetworkClassifier(self.new_dataset_name, saved_models_path, x_dim + z_dim, n_classes,
                                             hidden_dims=hidden_dims, dropout=dropout, lr=lr, wd=wd,
                                             device=device, figures_dir=figures_dir, seed=seed,
                                             batch_norm=batch_norm)

    @property
    def name(self) -> str:
        return f"network_use_z={self.use_z}"

    @property
    def classifier(self) -> ClassificationModel:
        return self._classifier


class CNNMaskEstimator(DataMaskEstimator):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, z_dim: int, hidden_dims: List[int] = None,
                 dropout: float = 0.1, batch_norm: bool = False, lr: float = 1e-3, wd: float = 0., device='cpu',
                 figures_dir=None, seed=0):
        super().__init__(dataset_name, x_dim, z_dim)
        n_classes = 2
        self.classifier = CNNClassifier(self.new_dataset_name, saved_models_path, x_dim, z_dim, n_classes,
                                        hidden_dims=hidden_dims, dropout=dropout, lr=lr, wd=wd,
                                        device=device, figures_dir=figures_dir, seed=seed,
                                        batch_norm=batch_norm)

    def forward(self, x, z) -> ClassProbabilities:
        return self.classifier.estimate_probabilities(x, z=z)

    def fit(self, x_train, z_train: Optional[torch.Tensor], deleted_train, x_val, z_val, deleted_val, epochs=1000,
            batch_size=64, n_wait=20,
            **kwargs):
        device = x_train.device
        super().fit(x_train, z_train, deleted_train, x_val, z_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)
        model_y_train = deleted_train
        model_d_train = torch.zeros(len(model_y_train), device=device)
        model_y_val = deleted_val
        model_d_val = torch.zeros(len(model_y_val), device=device)
        self.classifier.fit(x_train, model_y_train, model_d_train, x_val, model_y_val, model_d_val,
                            epochs, batch_size, n_wait,
                            z_train=z_train,
                            z_val=z_val)

    @property
    def name(self) -> str:
        return f"cnn_use_z={self.use_z}"


class XGBoostMaskEstimator(TabularDataMaskEstimator):

    def __init__(self, dataset_name, saved_models_path, x_dim: int,
                 z_dim: int, device='cpu',
                 seed=0):
        super().__init__(dataset_name, x_dim, z_dim)
        n_classes = 2
        self._classifier = XGBClassifier(self.new_dataset_name, saved_models_path, x_dim + z_dim, n_classes,
                                         device=device, seed=seed)

    @property
    def name(self) -> str:
        return f"xgb_use_z={self.use_z}"

    @property
    def classifier(self) -> ClassificationModel:
        return self._classifier


class RFMaskEstimator(TabularDataMaskEstimator):

    def __init__(self, dataset_name, saved_models_path, x_dim: int,
                 z_dim: int, device='cpu',
                 seed=0):
        super().__init__(dataset_name, x_dim, z_dim)
        n_classes = 2
        self._classifier = RFClassifier(self.new_dataset_name, saved_models_path, x_dim + z_dim, n_classes,
                                        device=device, seed=seed)

    @property
    def name(self) -> str:
        return f"rf_use_z={self.use_z}"

    @property
    def classifier(self) -> ClassificationModel:
        return self._classifier
