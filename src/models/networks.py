import functools
from typing import List
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from tqdm import tqdm


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


class BaseModel(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims=None, n_layers=3, dropout=0., bias=True, non_linearity='lrelu',
                 batch_norm=False, last_layer=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64] * n_layers

        n_layers = len(hidden_dims)
        if n_layers == 0:
            self.intermediate_layer = nn.Sequential()
            modules = [nn.Linear(in_dim, out_dim, bias=bias)]

        else:
            modules = [nn.Linear(in_dim, hidden_dims[0], bias=bias)]
            if dropout > 0:
                modules += [nn.Dropout(dropout)]
            if batch_norm:
                modules += [nn.BatchNorm1d(hidden_dims[0])]
            modules += [get_non_linearity(non_linearity)()]

            for i in range(n_layers - 1):
                modules += [nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias)]
                if batch_norm:
                    modules += [nn.BatchNorm1d(hidden_dims[i + 1])]
                modules += [get_non_linearity(non_linearity)()]
                if dropout > 0:
                    modules += [nn.Dropout(dropout)]

            self.intermediate_layer = nn.Sequential(*modules)

            modules += [nn.Linear(hidden_dims[-1], out_dim, bias=bias)]

        if last_layer is not None:
            modules += [last_layer()]

        self.network = nn.Sequential(*modules)

    def forward(self, x, **kwargs):
        return self.network(x)

    def change_requires_grad(self, new_val):
        for p in self.parameters():
            p.requires_grad = new_val

    def freeze(self):
        self.change_requires_grad(True)

    def unfreeze(self):
        self.change_requires_grad(False)


class CNN(nn.Module):
    def __init__(self, dataset_name, saved_models_path, x_dim: int, z_dim: int, out_dim: int,
                 hidden_dims: List[int] = None, dropout: float = 0.1, batch_norm: bool = False,
                 network_name=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if network_name is None:
            network_name = 'resnet18'  # default
        self.z_dim = z_dim
        cnn = self.get_network_from_name(network_name)
        # for param in cnn.parameters():
        #     param.requires_grad = False
        last_layers = list(cnn.children())[-1]
        if self.z_dim == 0:
            cnn.fc = nn.Sequential(
                nn.Linear(last_layers.in_features, out_dim)
            )
            self._network = cnn

        else:
            cnn_out_dim = hidden_dims[0] // 2
            base_model = BaseModel(cnn_out_dim + z_dim, out_dim, hidden_dims=hidden_dims, dropout=dropout,
                                   batch_norm=batch_norm)
            cnn.fc = nn.Linear(in_features=last_layers.in_features, out_features=cnn_out_dim, bias=True)
            self._network = cnn
            self.base_model = base_model

        self._network.train()
        self.is_train = True
        self.inference_batch_size = 32

    def forward(self, x, z: torch.Tensor = None, transform=None, **kwargs):
        if self.z_dim > 0 and z is None:
            raise Exception(f"cnn must get z for z_dim={self.z_dim}")
        batch_size = self.inference_batch_size
        requires_grad = x.requires_grad
        if len(x) > batch_size:
            x_pred = None
            for i in range(0, len(x), batch_size):
                start, end = i, min(i + batch_size, len(x))
                curr_x = x[start:end]
                if transform is not None:
                    curr_x = transform(curr_x)
                curr_x_pred = self._network(curr_x).squeeze(-1)
                if x_pred is None:
                    x_pred = curr_x_pred
                else:
                    x_pred = torch.cat([x_pred, curr_x_pred], dim=0)
                if not requires_grad:
                    x_pred = x_pred.detach()

        else:
            x_pred = self._network(x)
        # x_pred = self._network(x).squeeze()
        if self.z_dim > 0:
            return self.base_model(torch.cat([x_pred, z], dim=-1))
        else:
            return x_pred

    @staticmethod
    def get_network_from_name(network_name):
        if network_name == 'resnet50':
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if network_name == 'resnet18':
            return resnet18()
        else:
            raise Exception(f"does not know what network to return for: {network_name}")
