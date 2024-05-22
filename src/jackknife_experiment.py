import argparse
import ast
import copy
import warnings
from typing import List
import numpy as np
import torch
from calibration_schemes.AbstractCalibration import Calibration
from calibration_schemes.CQRCalibration import CQRCalibration
from calibration_schemes.OracleCQRCalibration import OracleCQRCalibration
from calibration_schemes.WeightedCalibration import WeightedCalibration
from calibration_schemes.PrivilegedConformalPrediction import PrivilegedConformalPrediction
from calibration_schemes.DummyCalibration import DummyCalibration
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_type import DataType
from data_utils.get_dataset_utils import get_regression_dataset, get_z_dim_from_data_name
from models.data_mask_estimators.OracleDataMasker import OracleDataMasker
from models.model_utils import ModelPrediction
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.data_mask_estimators.NetworkMaskEstimator import XGBoostMaskEstimator
from models.qr_models.XGBoostQR import XGBoostQR
from results_helper.regression_results_helper import RegressionResultsHelper
from utils import set_seeds
import matplotlib
from sys import platform

if platform not in ['win32', 'darwin']:
    matplotlib.use('Agg')

warnings.filterwarnings("ignore")


def parse_args_utils(args):
    args.hidden_dims = ast.literal_eval(args.hidden_dims)
    args.batch_norm = args.batch_norm > 0
    args.data_type = DataType.Real if args.data_type.lower() == 'real' else DataType.Synthetic
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device_name)
    print(f"device: {device_name}")
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default="real",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')
    parser.add_argument('--x_dim', type=int, default=15,
                        help='x dim of synthetic dataset')

    parser.add_argument('--dataset_name', type=str, default='missing_y_ihdp',
                        help='dataset to use')
    parser.add_argument('--data_path', type=str, default='datasets/real_data',
                        help='')
    parser.add_argument('--non_linearity', type=str, default="lrelu",
                        help='')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='')

    parser.add_argument('--data_size', type=int, default=30000,
                        help='')
    parser.add_argument('--hidden_dims', type=str, default='[32, 64, 64, 32]',
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='risk level')
    parser.add_argument('--bs', type=int, default=128,
                        help='batch size')
    parser.add_argument('--wait', type=int, default=200,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0,
                        help='weight decay')
    parser.add_argument('--base_results_save_dir', type=str, default="./results",
                        help="results save dir")

    parser.add_argument('--training_ratio', type=float, default=0.3,
                        help="fraction of samples used for test")
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                        help="fraction of samples used for validation")
    parser.add_argument('--calibration_ratio', type=float, default=0.,
                        help="fraction of samples used for validation")
    parser.add_argument('--epochs', type=int, default=5000,
                        help="number of epochs for offline training")
    parser.add_argument('--figures_dir', type=str, default='./figures',
                        help="figures_dir")
    parser.add_argument('--saved_models_path', type=str, default='./saved_models/jackknife',
                        help="saved_models_path")
    parser.add_argument('--batch_norm', type=int, default=0,
                        help="batch norm")
    args = parser.parse_args()
    args = parse_args_utils(args)
    return args



def get_calibration_schemes_from_params(dataset_name, x_dim, y_dim, z_dim, args, data_scaler, scaled_y_max: float,
                                        scaled_y_min: float,
                                        data_masker: DataCorruptionMasker,
                                        hidden_dims, dropout, batch_norm, lr, wd, device,
                                        saved_models_path: str, figures_dir: str, seed) -> List[Calibration]:
    alpha = args.alpha
    calibration_schemes = [DummyCalibration(alpha), CQRCalibration(alpha), CQRCalibration(alpha, ignore_masked=True),
                           OracleCQRCalibration(alpha)]

    def get_data_mask_estimators() -> List[DataMaskEstimator]:
        return [
            XGBoostMaskEstimator(dataset_name, saved_models_path, x_dim, z_dim,
                                 device=device,
                                 seed=seed),
            XGBoostMaskEstimator(dataset_name, saved_models_path, x_dim, 0,
                                 device=device,
                                 seed=seed),
            OracleDataMasker(data_scaler, data_masker, dataset_name, x_dim, z_dim),
        ]

    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            WeightedCalibration(CQRCalibration(alpha), alpha, dataset_name, data_scaler, data_mask_estimator))
    #
    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            PrivilegedConformalPrediction(CQRCalibration(alpha), alpha, dataset_name, data_scaler,
                                          data_mask_estimator))

    return calibration_schemes


def main(args=None):
    if args is None:
        args = parse_args()
    args.z_dim = get_z_dim_from_data_name(args.dataset_name)
    print(f"starting seed: {args.seed} data: {args.dataset_name}")
    set_seeds(args.seed)
    dataset = get_regression_dataset(args)
    results_helper = RegressionResultsHelper(args.base_results_save_dir, args.seed)
    alpha = args.alpha

    def get_calibration_schemes():
        return get_calibration_schemes_from_params(dataset.dataset_name, dataset.x_dim, dataset.y_dim,
                                                   dataset.z_dim,
                                                   args,
                                                   dataset.scaler,
                                                   dataset.scaled_y_max, dataset.scaled_y_min,
                                                   dataset.data_masker,
                                                   args.hidden_dims, args.dropout, args.batch_norm,
                                                   args.lr, args.wd,
                                                   args.device,
                                                   args.saved_models_path,
                                                   args.figures_dir,
                                                   args.seed)

    calibration_schemes: List[Calibration] = get_calibration_schemes()

    x = torch.cat([dataset.x_train, dataset.x_val, dataset.x_cal])
    y = torch.cat([dataset.y_train, dataset.y_val, dataset.y_cal])
    z = torch.cat([dataset.z_train, dataset.z_val, dataset.z_cal])
    d = torch.cat([dataset.deleted_train, dataset.deleted_val, dataset.deleted_cal])
    n = x.shape[0]
    test_model_prediction_list: List[ModelPrediction] = []
    cal_model_prediction_list: List[ModelPrediction] = []
    for i in range(n):
        permutation = np.random.permutation(n)
        permutation = permutation[permutation != i]
        assert i not in permutation.tolist()
        train_idx = permutation[int(len(permutation) * args.validation_ratio):]
        val_idx = permutation[:int(len(permutation) * args.validation_ratio)]

        x_train = x[train_idx]
        y_train = y[train_idx]
        z_train = z[train_idx]
        d_train = d[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        z_val = z[val_idx]
        d_val = d[val_idx]
        x_test = dataset.x_test
        x_cal = x[i].unsqueeze(0)
        y_cal = y[i].unsqueeze(0)
        z_cal = z[i].unsqueeze(0)
        d_cal = d[i].unsqueeze(0)
        model = XGBoostQR(f'{dataset.dataset_name}_{i}', args.saved_models_path, seed=args.seed, alpha=alpha)
        model.fit(x_train, y_train, d_train, x_val, y_val, d_val, args.epochs, args.bs, args.wait)
        model.eval()
        test_model_prediction_list += [model.construct_uncalibrated_intervals(x_test)]
        cal_uncalibrated_intervals = model.construct_uncalibrated_intervals(x_cal)
        cal_model_prediction_list += [cal_uncalibrated_intervals]

    for calibration_scheme in calibration_schemes:
        calibration_scheme.fit(dataset.x_train, dataset.y_train, dataset.z_train, dataset.deleted_train,
                               dataset.x_val, dataset.y_val, dataset.z_val, dataset.deleted_val,
                               epochs=args.epochs, batch_size=args.bs, n_wait=args.wait)
        full_y_cal = torch.cat([dataset.full_y_train, dataset.full_y_val, dataset.full_y_cal])
        test_uncertainty_sets = calibration_scheme.jackknife_plus_construct_uncertainty_set_from_scores(x, y, z, d,
                                                                                                        cal_model_prediction_list,
                                                                                                        dataset.x_test,
                                                                                                        test_model_prediction_list,
                                                                                                        z_test=dataset.z_test,
                                                                                                        full_y_cal=full_y_cal)
        results_helper.save_performance_metrics(
            None, None,
            test_model_prediction_list[0],
            test_uncertainty_sets,
            dataset,
            model,
            calibration_scheme)


if __name__ == '__main__':
    args = parse_args()
    for seed in range(0, 50):
        args.seed = seed
        main(args)
