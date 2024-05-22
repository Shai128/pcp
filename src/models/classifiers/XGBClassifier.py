import os
from typing import List

import torch
from torch import nn
from models.ClassificationModel import ClassificationModel, ClassProbabilities
from models.abstract_models.LearningModel import LearningModel
import xgboost as xgb


class XGBClassifier(ClassificationModel, LearningModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, n_classes: int, device='cpu',
                 learning_rate: float = None, max_depth: int = None, n_estimators: int = 100, seed=0):
        ClassificationModel.__init__(self)
        LearningModel.__init__(self, dataset_name, saved_models_path, seed)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_boost_round = 10
        self.early_stopping_rounds = None
        self.n_estimators = n_estimators
        self.num_class = n_classes
        self.model = xgb.Booster()

    def fit_xy_aux(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64,
                   n_wait=20, **kwargs):
        Xy = xgb.QuantileDMatrix(x_train.detach().cpu().numpy(), y_train.detach().cpu().numpy())
        Xy_val = xgb.QuantileDMatrix(x_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
        self.model = xgb.train(
            {
                "objective": 'multi:softprob',
                "num_class": self.num_class,
                # Let's try not to overfit.
                # "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "n_estimators": self.n_estimators
            },
            Xy,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds,
            # The evaluation result is a weighted average across multiple quantiles.
            evals=[(Xy, "Train"), (Xy_val, "Validation")],
            verbose_eval=False
        )

    def store_model(self):
        self.model.save_model(os.path.join(self.get_model_save_dir(), f"seed={self.seed}.pth"))

    def load_model(self):
        self.model.load_model(os.path.join(self.get_model_save_dir(), f"seed={self.seed}.pth"))

    @property
    def name(self) -> str:
        return "xgb_classifier"

    def estimate_probabilities(self, x: torch.Tensor) -> ClassProbabilities:
        probabilities = torch.Tensor(self.model.inplace_predict(x.detach().cpu().numpy())).to(x.device)
        return ClassProbabilities(probabilities)


    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)

    def eval(self):
        pass
