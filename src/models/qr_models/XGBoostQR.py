import abc
import os.path
from typing import Dict

import numpy as np
import torch

from models.abstract_models.AbstractModel import Model
from models.abstract_models.LearningModel import LearningModel
import xgboost as xgb

from models.qr_models.PredictionIntervalModel import PredictionIntervalModel, PredictionIntervals
from utils import create_folder_if_it_doesnt_exist


class XGBoostQR(LearningModel, PredictionIntervalModel):

    def __init__(self, dataset_name: str, saved_models_path: str, seed: int, alpha: float, learning_rate: float = None,
                 max_depth: int = None, n_estimators: int = 100):
        LearningModel.__init__(self, dataset_name, saved_models_path, seed)
        self.alpha = alpha
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.num_boost_round = 10
        self.early_stopping_rounds = None
        self.xgb_low = xgb.Booster()
        self.xgb_high = xgb.Booster()

    def fit_xy_aux(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64,
                   n_wait=20,
                   **kwargs):
        alpha_low = self.alpha / 2
        alpha_high = 1 - self.alpha / 2
        Xy = xgb.QuantileDMatrix(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
        Xy_val = xgb.QuantileDMatrix(x_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
        evals_result: Dict[str, Dict] = {}
        general_params = {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
        }
        self.xgb_low = xgb.train(
            {

                "quantile_alpha": alpha_low,
                **general_params
            },
            Xy,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            # The evaluation result is a weighted average across multiple quantiles.
            evals=[(Xy, "Train"), (Xy_val, "Validation")],
            evals_result=evals_result,
            verbose_eval=False
        )
        # low_pred_val = self.xgb_low.inplace_predict(x_val.detach().cpu().numpy())
        # res = (y_val.detach().cpu().numpy() <= low_pred_val).astype(float).mean()

        self.xgb_high = xgb.train(
            {
                "quantile_alpha": alpha_high,
                **general_params
            },
            Xy,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            # The evaluation result is a weighted average across multiple quantiles.
            evals=[(Xy, "Train"), (Xy_val, "Validation")],
            evals_result=evals_result,
            verbose_eval=False
        )
        # high_pred_val = self.xgb_high.inplace_predict(x_val.detach().cpu().numpy())
        # res = (y_val.detach().cpu().numpy() <= high_pred_val).astype(float).mean()
        # print()

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)

    def construct_uncalibrated_intervals(self, x: torch.Tensor) -> PredictionIntervals:
        device = x.device
        high_pred = self.xgb_high.inplace_predict(x.detach().cpu().numpy())
        low_pred = self.xgb_low.inplace_predict(x.detach().cpu().numpy())
        intervals = torch.from_numpy(np.concatenate([low_pred[..., None], high_pred[..., None]], axis=-1)).to(device)
        return PredictionIntervals(intervals)

    def get_model_save_dir(self) -> str:
        return os.path.join(self.saved_models_path, self.dataset_name, self.save_name)

    def get_model_save_path(self) -> str:
        return os.path.join(self.get_model_save_dir(), f"low_seed={self.seed}.pth")

    def store_model(self):
        create_folder_if_it_doesnt_exist(self.get_model_save_dir())
        self.xgb_low.save_model(os.path.join(self.get_model_save_dir(), f"low_seed={self.seed}.pth"))
        self.xgb_high.save_model(os.path.join(self.get_model_save_dir(), f"high_seed={self.seed}.pth"))

    def load_model(self):
        self.xgb_low.load_model(os.path.join(self.get_model_save_dir(), f"low_seed={self.seed}.pth"))
        self.xgb_high.load_model(os.path.join(self.get_model_save_dir(), f"high_seed={self.seed}.pth"))

    @property
    def name(self) -> str:
        return "xgb_qr"
