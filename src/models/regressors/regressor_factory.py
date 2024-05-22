from enum import Enum, auto

from models.regressors.DMLRegressor import DMLRegressor
from models.regressors.FullRegressor import FullRegressor
from models.regressors.FullRegressorWithLinear import FullRegressorWithLinear
from models.regressors.LinearRegressor import LinearRegressor
from models.regressors.MeanRegressor import MeanRegressor
from models.regressors.PartiallyLinearIndependentRegressor import PartiallyLinearIndependentRegressor
from models.regressors.PartiallyLinearRegressor import PartiallyLinearRegressor


class RegressorType(Enum):
    Linear = auto()
    PartiallyLinear = auto()
    DML = auto()
    Full = auto()
    FullWithLinearity = auto()
    IndependentPartiallyLinear = auto()


class RegressorFactory:
    def __init__(self, dataset_name: str, saved_models_path: str, figures_dir: str, seed: int, x_dim: int, y_dim: int,
                 z_dim: int,
                 hidden_dims, batch_norm: bool, dropout: float, lr: float, wd: float, device):
        self.dataset_name = dataset_name
        self.saved_models_path = saved_models_path
        self.figures_dir = figures_dir
        self.seed = seed
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.lr = lr
        self.wd = wd
        self.device = device

    def generate_regressor(self, regressor_type: RegressorType) -> MeanRegressor:
        if regressor_type == RegressorType.Linear:
            return LinearRegressor(self.dataset_name, self.saved_models_path, figures_dir=self.figures_dir,
                                   seed=self.seed)
        elif regressor_type == RegressorType.PartiallyLinear:
            return PartiallyLinearRegressor(self.dataset_name, self.saved_models_path, self.x_dim, self.z_dim,
                                            self.hidden_dims, self.dropout,
                                            self.batch_norm, self.lr, self.wd, self.device,
                                            figures_dir=self.figures_dir, seed=self.seed)
        elif regressor_type == RegressorType.DML:
            return DMLRegressor(self.dataset_name, self.saved_models_path, self.x_dim, self.y_dim,
                                self.hidden_dims,
                                self.dropout,
                                self.batch_norm, self.lr, self.wd, self.device, figures_dir=self.figures_dir,
                                seed=self.seed)
        elif regressor_type == RegressorType.Full:
            return FullRegressor(self.dataset_name, self.saved_models_path, self.x_dim, self.z_dim, self.hidden_dims,
                                 self.dropout, self.batch_norm, self.lr, self.wd, self.device,
                                 figures_dir=self.figures_dir,
                                 seed=self.seed)
        elif regressor_type == RegressorType.FullWithLinearity:
            return FullRegressorWithLinear(self.dataset_name, self.saved_models_path, self.x_dim,
                                           self.z_dim,
                                           self.hidden_dims,
                                           self.dropout,
                                           self.batch_norm, self.lr, self.wd, self.device,
                                           figures_dir=self.figures_dir,
                                           seed=self.seed)
        elif regressor_type == RegressorType.IndependentPartiallyLinear:
            return PartiallyLinearIndependentRegressor(self.dataset_name,
                                                       self.saved_models_path, self.x_dim,
                                                       self.z_dim,
                                                       self.hidden_dims, self.dropout,
                                                       self.batch_norm, self.lr, self.wd,
                                                       self.device,
                                                       figures_dir=self.figures_dir,
                                                       seed=self.seed)
        else:
            raise Exception(f"invalid regressor type: {regressor_type.name}")
