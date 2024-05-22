import functools
from typing import List
from torch.nn.modules.module import T
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from tqdm import tqdm

from models.ClassificationModel import ClassificationModel, ClassProbabilities
from models.abstract_models.NetworkLearningModel import NetworkLearningModel
import torchvision.transforms as transforms

from models.cnn_utils import test_transform, train_transform
from models.networks import BaseModel, CNN


class CNNClassifier(ClassificationModel, NetworkLearningModel):

    def __init__(self, dataset_name, saved_models_path, x_dim: int, z_dim: int,
                 n_classes: int,
                 hidden_dims: List[int] = None, dropout: float = 0.1,
                 batch_norm: bool = False,
                 lr: float = 1e-3, wd: float = 0., device='cpu', figures_dir=None,
                 seed=0,
                 network_name=None):
        NetworkLearningModel.__init__(self, dataset_name, saved_models_path, figures_dir, seed)
        ClassificationModel.__init__(self)

        self.z_dim = z_dim
        self._network = CNN(dataset_name, saved_models_path, x_dim, z_dim,
                            n_classes, hidden_dims, dropout, batch_norm, network_name=network_name)
        self._network = self._network.to(device)
        self._network.train()
        self._optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=wd)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.wd = wd
        self.device = device
        self.n_classes = n_classes
        self.test_transform = test_transform
        self.train_transform = train_transform

    def loss(self, y: torch.Tensor, prediction, d, epoch):
        return self.cross_entropy_loss(prediction, y.long().squeeze())

    def predict(self, x, z=None, transform=None):
        model_pred = self._network(x, z=z, transform=transform)
        return torch.softmax(model_pred, dim=-1)

    @property
    def name(self) -> str:
        return "cnn"

    def estimate_probabilities(self, x: torch.Tensor, z: torch.Tensor = None) -> ClassProbabilities:
        pred = self.predict(x, z=z, transform=self.test_transform)
        return ClassProbabilities(pred)

    def fit(self, x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=1000, batch_size=64, n_wait=20,
            **kwargs):
        self.fit_xy(x_train, y_train, deleted_train, x_val, y_val, deleted_val, epochs=epochs, batch_size=batch_size,
                    n_wait=n_wait, **kwargs,
                    train_transform=self.train_transform,
                    test_transform=self.test_transform,
                    )

    def train(self: T, mode: bool = True) -> T:
        return self._network.train(mode)

    def eval(self):
        self._network.eval()
