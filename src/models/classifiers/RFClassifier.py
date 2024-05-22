import os
from typing import List

import torch
from torch import nn
from models.ClassificationModel import ClassificationModel, ClassProbabilities
from models.abstract_models.LearningModel import LearningModel
from sklearn.ensemble import RandomForestClassifier
import joblib


class RFClassifier(ClassificationModel, LearningModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, n_classes: int, device='cpu',
                 max_depth: int = None, n_estimators: int = 100, seed=0):
        ClassificationModel.__init__(self)
        LearningModel.__init__(self, dataset_name, saved_models_path, seed)
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.num_class = n_classes
        self.clf = RandomForestClassifier(max_depth=max_depth,
                                          n_estimators=n_estimators,
                                          random_state=seed)

    def fit_xy_aux(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64,
                   n_wait=20, **kwargs):
        new_x_train = torch.cat([x_train, x_val], dim=0).detach().cpu().numpy()
        new_y_train = torch.cat([y_train, y_val], dim=0).detach().cpu().numpy()
        self.clf.fit(new_x_train, new_y_train)

    @property
    def name(self) -> str:
        return "rf_classifier"

    def estimate_probabilities(self, x: torch.Tensor) -> ClassProbabilities:
        probabilities = torch.Tensor(self.clf.predict_proba(x.detach().cpu().numpy())).to(x.device)
        return ClassProbabilities(probabilities)

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)

    def eval(self):
        pass

    def store_model(self):
        joblib.dump(self.clf, self.get_model_save_path())

    def load_model(self):
        self.clf = joblib.load(self.get_model_save_path())
