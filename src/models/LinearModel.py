import abc

import sklearn.linear_model
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression

from models.abstract_models.NetworkLearningModel import NetworkLearningModel


class SKLearnModelWrapper(NetworkLearningModel):
    def __init__(self, dataset_name, saved_models_path, figures_dir: str, seed):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)

    def fit(self, x_train, y_train, deleted_train=None, x_val=None, y_val=None, deleted_val=None):
        if deleted_train is None:
            deleted_train = torch.zeros(len(x_train)).to(x_train.device).bool()
        if x_val is not None and y_val is not None and deleted_val is not None:
            deleted_train = torch.cat([deleted_train, deleted_val], dim=0)
            x_train = torch.cat([x_train, x_val], dim=0)
            y_train = torch.cat([y_train, y_val], dim=0)
        deleted_train = deleted_train.detach().cpu()
        x_train = x_train.detach().cpu()[~deleted_train]
        y_train = y_train.detach().cpu()[~deleted_train]
        if len(x_train.shape) == 1:
            x_train = x_train.unsqueeze(-1)
        self.model.fit(x_train, y_train)

    def loss(self, y, prediction, d, epoch):
        pass

    def predict(self, x, **kwargs):
        device = x.device
        if len(x.shape) == 1:
            x = x.unsqueeze(-1)
        return torch.Tensor(self.model.predict(x.detach().cpu())).to(device)

    @property
    def name(self) -> str:
        return "linear"

    @property
    @abc.abstractmethod
    def model(self) -> sklearn.linear_model._base.LinearModel:
        pass


class LinearModel(SKLearnModelWrapper):

    def __init__(self, dataset_name, saved_models_path, figures_dir: str, seed):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.reg = LinearRegression()

    @property
    def name(self) -> str:
        return "linear"

    @property
    def model(self) -> sklearn.linear_model._base.LinearModel:
        return self.reg

    def get_coeff(self):
        return torch.Tensor(self.reg.coef_)

class LogisticLinearModel(SKLearnModelWrapper):

    def __init__(self, dataset_name, saved_models_path, figures_dir: str, seed):
        super().__init__(dataset_name, saved_models_path, figures_dir=figures_dir, seed=seed)
        self.reg = LogisticRegression()

    @property
    def name(self) -> str:
        return "logistic_linear"

    @property
    def model(self) -> sklearn.linear_model._base.LinearModel:
        return self.reg

    def estimate_probabilities(self, x: torch.Tensor):
        device = x.device
        return torch.Tensor(self.reg.predict_proba(x.detach().cpu())).to(device)