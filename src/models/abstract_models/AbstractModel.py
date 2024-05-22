import abc
import torch
from torch import nn


class Model:
    def __init__(self):
        super().__init__()

    # @abc.abstractmethod
    # def fit(self, x_train, y_train, z_train, deleted_train, x_val, y_val, z_val, deleted_val, epochs=1000, batch_size=64, n_wait=20, **kwargs):
    #     pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def compute_performance(self, x_test, z_test, full_y_test, deleted_test) -> dict:
        return {}
