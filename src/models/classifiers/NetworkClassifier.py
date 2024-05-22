from typing import List

import torch
from torch import nn

from models.ClassificationModel import ClassificationModel, ClassProbabilities
from models.abstract_models.NetworkLearningModel import NetworkLearningModel
from models.networks import BaseModel
from torchvision.models import resnet50


# img = Image.open("test/assets/encode_jpeg/grace_hopper_517x606.jpg")

# Step 1: Initialize model

class NetworkClassifier(ClassificationModel, NetworkLearningModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, n_classes: int,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False,
                 lr: float = 1e-3, wd: float = 0., device='cpu', figures_dir=None,
                 seed=0,
                 network_name=None):
        NetworkLearningModel.__init__(self, dataset_name, saved_models_path, figures_dir, seed)
        ClassificationModel.__init__(self)

        if hidden_dims is None:
            hidden_dims = [32, 64, 64, 32]
        self._network = BaseModel(x_dim, n_classes, hidden_dims=hidden_dims, dropout=dropout,
                                  batch_norm=batch_norm).to(device)

        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.wd = wd
        self.device = device
        self.n_classes = n_classes

    def loss(self, y: torch.Tensor, prediction, d, epoch):
        return self.cross_entropy_loss(prediction, y.long().squeeze())

    def predict(self, x, **kwargs):
        return torch.softmax(self.network(x).squeeze(), dim=-1)

    @property
    def name(self) -> str:
        return "simple_classifier"

    def estimate_probabilities(self, x: torch.Tensor) -> ClassProbabilities:
        return ClassProbabilities(torch.softmax(self.network.forward(x), -1))

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs)

    def eval(self):
        self._network.eval()
