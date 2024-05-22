from __future__ import annotations

import abc
from typing import List

import torch


class ModelPrediction(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self, item):
        pass


class UncertaintySets(abc.ABC):

    @abc.abstractmethod
    def __getitem__(self, item):
        pass

    @abc.abstractmethod
    def contains(self, y):
        pass

    @abc.abstractmethod
    def union(self, other_set: UncertaintySets) -> UncertaintySets:
        pass


def construct_interval_from_pred(pred: torch.Tensor):
    """

    :param pred: tensor of size: n x (2*p), where p is the output size. pred[i *k] is the lower bound for the k-th
    element and pred[i *k + 1] is the upper bound for the k-th element
    :return: interval for each of the n samples, of size: n x p x 2
    """
    lower_q = pred[:, 0::2]
    upper_q = pred[:, 1::2]
    interval = torch.cat([lower_q.unsqueeze(-1), upper_q.unsqueeze(-1)], dim=-1)
    return interval


def batch_pinball_loss(quantile_level, quantile, y):
    diff = quantile - y
    mask = (diff.ge(0).float() - quantile_level).detach()

    return (mask * diff).mean()


def two_dimensional_pinball_loss(y, prediction, alpha):
    # y = y.squeeze()
    n = len(y)
    lower_quantile_level = torch.Tensor([alpha / 2]).repeat(n).to(y.device)
    upper_quantile_level = torch.Tensor([1 - alpha / 2]).repeat(n).to(y.device)
    lower_pred = prediction[:, 0::2]
    upper_pred = prediction[:, 1::2]
    y_rep = torch.cat([y, y])
    pred_rep = torch.cat([lower_pred, upper_pred])
    quantile_level_rep = torch.cat([lower_quantile_level, upper_quantile_level]).unsqueeze(-1).repeat(1, y.shape[1])
    diff = pred_rep - y_rep
    mask = (diff.ge(0).float() - quantile_level_rep).detach()
    return (mask * diff).mean()
