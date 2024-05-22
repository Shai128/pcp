import os
import sys
from typing import Union

import numpy as np
import sklearn.cluster
import torch
import random
from sklearn.cluster import KMeans



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_seed():
    return np.random.randint(0, 2 ** 31)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


def weighted_quantile(values: torch.Tensor, quantiles: torch.Tensor, sample_weight: Union[torch.Tensor, None] = None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    if sample_weight is None:
        sample_weight = torch.ones(len(values), device=values.device)
    if type(quantiles) == float:
        quantiles = torch.Tensor([quantiles]).to(values.device)
    assert torch.all(quantiles >= 0) and torch.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = torch.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = torch.cumsum(sample_weight, dim=0) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0].item()
        weighted_quantiles /= weighted_quantiles[-1].item()
    else:
        weighted_quantiles /= torch.sum(sample_weight)
    return interp(quantiles, weighted_quantiles, values)


def corr(x, y):
    # if len(x) < len(y):
    #     idx = np.random.permutation(len(y))[:len(x)]
    #     y = y[idx]
    # elif len(y) < len(x):
    #     idx = np.random.permutation(len(x))[:len(y)]
    #     x = x[idx]
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


def filter_missing_values(x, y, deleted):
    x, y = x[~deleted], y[~deleted]
    deleted = torch.zeros_like(x)
    return x, y, deleted


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1)).squeeze()
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    if len(y.shape) == 1:
        y = y.unsqueeze(-1)
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m, device=K.device) - 1.0 / m * torch.ones((m, m), device=K.device)
    H = H.float()
    hsic = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    if hsic.isnan().any() or hsic.isinf().any():
        hsic = torch.Tensor([0]).squeeze().to(x.device)
    if (hsic < 0).any():
        hsic = torch.Tensor([0]).squeeze().to(x.device)
    return hsic


def MMD(x, y, kernel='multiscale'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    device = x.device
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    if len(y.shape) == 1:
        y = y.unsqueeze(-1)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def create_folder_if_it_doesnt_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def compute_ece(y, prediction, n_bins=10):
    predicted_probability = prediction.max(dim=-1)[0]
    predicted_index = prediction.argmax(dim=-1)

    acc, conf = torch.zeros(n_bins), torch.zeros(n_bins)
    Bm = torch.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        bin_idx = (predicted_probability > a) & (predicted_probability <= b)
        Bm[m] = bin_idx.int().sum().item()
        correct_in_bin = (predicted_index == y)[bin_idx]
        acc[m] = correct_in_bin.float().mean().item()
        conf[m] = predicted_probability[bin_idx].float().mean().item()

    Bm = np.array(Bm)
    acc = np.array(acc)[Bm > 0]
    conf = np.array(conf)[Bm > 0]
    Bm = Bm[Bm > 0]
    ece = 100 * np.abs(Bm * (acc - conf)).sum() / Bm.sum()
    return ece


# over calibration error
def compute_oce(y, prediction, n_clusters=None):
    n = y.shape[0]
    n_classes = prediction.shape[-1]
    if n_clusters is None:
        if n < 2000:
            print("low n")
            cluster_size = 80
        elif n < 5000:
            cluster_size = 150
        else:
            cluster_size = 300

        n_clusters = n // cluster_size

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    print("fitting kmeans")
    kmeans.fit(prediction.detach().cpu().numpy())
    clusters = kmeans.labels_
    y_one_hot = torch.zeros_like(prediction)
    y_one_hot[range(n), y.long()] = 1
    over_condifences = []
    for cluster in range(n_clusters):
        cluster_size = torch.Tensor(cluster == clusters).float().sum().item()
        estimated_probabilities = prediction[cluster == clusters].mean(dim=0)
        real_probabilities = y_one_hot[cluster == clusters].mean(dim=0)
        diff = estimated_probabilities - real_probabilities
        high_probability_idx = real_probabilities >= 1 / n_classes
        over_condifence = diff.clone()
        over_condifence[high_probability_idx] = torch.clamp(over_condifence[high_probability_idx], min=0)
        over_condifence[~high_probability_idx] = torch.clamp(over_condifence[~high_probability_idx], max=0)
        over_condifence = over_condifence.abs()
        over_condifences += [over_condifence.sum().item() * cluster_size / n]

    oce = 100 * np.sum(over_condifences)
    return oce


def compute_sample_entropy(prediction: torch.Tensor):
    return (-prediction * torch.log2(prediction)).sum(dim=-1)


def compute_entropy_deleted_correlation(prediction: torch.Tensor, deleted: torch.Tensor):
    entropy = compute_sample_entropy(prediction)
    return corr(entropy, deleted.float()).item()


def compute_average_entropy(prediction: torch.Tensor):
    return compute_sample_entropy(prediction).mean().item()


def compute_model_certainty(prediction: torch.Tensor):
    n_classes = prediction.shape[-1]
    most_certain_prediction = torch.zeros(n_classes)
    most_certain_prediction[0] = 1
    best_certainty = (most_certain_prediction - 1 / n_classes).abs().mean().item()
    certainty = (prediction - 1 / n_classes).abs().mean(dim=-1).mean().item() / best_certainty
    return certainty * 100


def interp1d_func(x, x0, x1, y0, y1):
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def batch_interp1d(x: torch.Tensor, y: torch.Tensor, a: float = None, b: float = None):
    if a is None or b is None:
        fill_value = 'extrapolate'
    else:
        fill_value = (a, b)

    def interp(desired_x):
        # desired_x = np.random.rand(3, 100) * 30 - 5
        desired_x = desired_x.to(x.device)
        if len(desired_x.shape) != 2 or desired_x.shape[0] != x.shape[0]:
            raise Exception(f"the shape of the input vector should be ({x.shape[0]},m), but got {desired_x.shape}")
        desired_x, _ = desired_x.sort()
        desired_x_rep = desired_x.unsqueeze(-1).repeat(1, 1, x.shape[-1] - 1)
        x_rep = x.unsqueeze(1).repeat(1, desired_x.shape[1], 1)
        relevant_idx = torch.stack(
            ((x_rep[:, :, :-1] <= desired_x_rep) & (desired_x_rep <= x_rep[:, :, 1:])).nonzero(as_tuple=True))

        x0 = x[relevant_idx[0], relevant_idx[2]]
        y0 = y[relevant_idx[0], relevant_idx[2]]
        x1 = x[relevant_idx[0], relevant_idx[2] + 1]
        y1 = y[relevant_idx[0], relevant_idx[2] + 1]
        desired_x_in_interpolation_range = desired_x[relevant_idx[0], relevant_idx[1]]
        res = torch.zeros_like(desired_x)
        res[relevant_idx[0], relevant_idx[1]] = interp1d_func(desired_x_in_interpolation_range, x0, x1, y0, y1)
        if fill_value == 'extrapolate':
            idx = (desired_x < x[:, 0, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], 0], x[idx[0], 1]
            y0, y1 = y[idx[0], 0], y[idx[0], 1]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

            idx = (desired_x > x[:, -1, None]).nonzero(as_tuple=True)
            x0, x1 = x[idx[0], -1], x[idx[0], -2]
            y0, y1 = y[idx[0], -1], y[idx[0], -2]
            res[idx[0], idx[1]] = interp1d_func(desired_x[idx[0], idx[1]], x0, x1, y0, y1)

        else:
            a, b = fill_value
            res[desired_x < x[:, 0, None]] = a
            res[desired_x > x[:, -1, None]] = b
        return res

    return interp


def batch_estim_dist(quantiles: torch.Tensor, percentiles: torch.Tensor, y_min, y_max, smooth_tails, tau,
                     extrapolate_quantiles=False):
    """ Estimate CDF from list of quantiles, with smoothing """
    device = quantiles.device
    noise = torch.rand_like(quantiles) * 1e-8
    noise_monotone, _ = torch.sort(noise)
    quantiles = quantiles + noise_monotone
    assert len(percentiles.shape) == 1 and len(quantiles.shape) == 2 and quantiles.shape[1] == percentiles.shape[0]
    percentiles = percentiles.unsqueeze(0).repeat(quantiles.shape[0], 1)

    # Smooth tails
    cdf = batch_interp1d(quantiles, percentiles, 0.0, 1.0)
    if extrapolate_quantiles:
        inv_cdf = batch_interp1d(percentiles, quantiles)
        return cdf, inv_cdf
    inv_cdf = batch_interp1d(percentiles, quantiles, y_min, y_max)

    if smooth_tails:
        # Uniform smoothing of tails
        quantiles_smooth = quantiles
        tau_lo = torch.ones(quantiles.shape[0], 1, device=device) * tau
        tau_hi = torch.ones(quantiles.shape[0], 1, device=device) * (1 - tau)
        q_lo = inv_cdf(tau_lo)
        q_hi = inv_cdf(tau_hi)
        idx_lo = torch.where(percentiles < tau_lo)[0]
        idx_hi = torch.where(percentiles > tau_hi)[0]
        if len(idx_lo) > 0:
            quantiles_smooth[idx_lo] = torch.linspace(quantiles[0], q_lo, steps=len(idx_lo), device=device)
        if len(idx_hi) > 0:
            quantiles_smooth[idx_hi] = torch.linspace(q_hi, quantiles[-1], steps=len(idx_hi), device=device)

        cdf = batch_interp1d(quantiles_smooth, percentiles, 0.0, 1.0)

    # Standardize
    breaks = torch.linspace(y_min, y_max, steps=1000, device=device).unsqueeze(0).repeat(quantiles.shape[0], 1)
    cdf_hat = cdf(breaks)
    f_hat = torch.diff(cdf_hat)
    f_hat = (f_hat + 1e-10) / (torch.sum(f_hat + 1e-10, dim=-1)).reshape((f_hat.shape[0], 1))
    cumsum = torch.cumsum(f_hat, dim=-1)
    cdf_hat = torch.cat([torch.zeros_like(cumsum)[:, 0:1], cumsum], dim=-1)
    cdf = batch_interp1d(breaks, cdf_hat, 0.0, 1.0)
    inv_cdf = batch_interp1d(cdf_hat, breaks, y_min, y_max)

    return cdf, inv_cdf

