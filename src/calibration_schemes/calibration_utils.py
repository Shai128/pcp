import numpy as np
import torch
from typing import Union

def compute_cqr_scores(y, intervals):
    return torch.max(intervals[:, 0] - y, y - intervals[:, 1])


def multi_dimensional_cqr_calibration(y_cal, deleted_cal, cal_interval, alpha):
    if len(y_cal.shape) == 1:
        y_cal = y_cal.unsqueeze(-1)
    deleted_cal = deleted_cal.squeeze()
    if len(cal_interval.shape) == 2:
        cal_interval = cal_interval.unsqueeze(1)
    Qs = torch.zeros(y_cal.shape[-1], device=y_cal.device)
    for k in range(y_cal.shape[-1]):
        cal_scores = torch.max(cal_interval[:, k, 0] - y_cal[:, k], y_cal[:, k] - cal_interval[:, k, 1])
        cal_scores = cal_scores[~deleted_cal]
        Qs[k] = torch.quantile(cal_scores, q=1 - alpha + (1 / (len(cal_scores) + 1)))
    return Qs

