from typing import List

import torch
from torch import nn

from models.ClassificationModel import ClassificationModel, ClassProbabilities
from models.networks import BaseModel
from utils import corr


class IndependentNetworkClassifier(ClassificationModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, n_classes: int,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False,
                 lr: float = 1e-3, wd: float = 0., device='cpu', figures_dir=None,
                 seed=0):
        ClassificationModel.__init__(self, dataset_name, saved_models_path, figures_dir, seed)

        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        self._network = BaseModel(x_dim, n_classes, hidden_dims=hidden_dims, dropout=dropout,
                                  batch_norm=batch_norm).to(device)
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.wd = wd

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            train_uncalibrated_intervals=None,
            val_uncalibrated_intervals=None, **kwargs):
        batch_size = 1024
        super().fit(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs, batch_size, n_wait)

    def loss(self, y: torch.Tensor, prediction, d, epoch):
        cross_entropy_loss = self.cross_entropy_loss(prediction, y.long()[:, 1])
        predicted_proba = torch.softmax(prediction, dim=-1)
        entropies = (-predicted_proba*torch.log2(predicted_proba)).sum(dim=-1)
        if torch.any(d):
            dependence_loss = corr(d.float(), entropies).abs()
        else:
            dependence_loss = 0
        return cross_entropy_loss + dependence_loss

    def predict(self, x):
        return self.network(x).squeeze()

    @property
    def name(self) -> str:
        return "independent_classifier"

    def estimate_probabilities(self, x: torch.Tensor) -> ClassProbabilities:
        return ClassProbabilities(torch.softmax(self.network.forward(x), -1))
