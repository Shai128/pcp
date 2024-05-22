import argparse
import ast
import copy
import warnings
from typing import List

import torch
import traceback
from calibration_schemes.AbstractCalibration import Calibration
from calibration_schemes.CQRCalibration import CQRCalibration
from calibration_schemes.WeightedCalibration import WeightedCalibration
from calibration_schemes.PrivilegedConformalPrediction import PrivilegedConformalPrediction
from calibration_schemes.TwoStagedConformalPrediction import ConservativeWeightedCalibration
from calibration_schemes.DummyCalibration import DummyCalibration
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker
from data_utils.data_type import DataType
from data_utils.datasets.dataset import Dataset
from data_utils.get_dataset_utils import get_regression_dataset, get_z_dim_from_data_name
from get_model_utils import get_proxy_qr_model, get_data_learning_mask_estimator, is_data_for_xgboost
from models.data_mask_estimators.OracleDataMasker import OracleDataMasker
from models.qr_models.PredictionIntervalModel import PredictionIntervalModel
from models.qr_models.QuantileRegression import QuantileRegression
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.data_mask_estimators.NetworkMaskEstimator import NetworkMaskEstimator, XGBoostMaskEstimator
from models.qr_models.XGBoostQR import XGBoostQR
from models.regressors.regressor_factory import RegressorType, RegressorFactory
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
    args.oracles = args.oracles > 0
    print(f"device: {device_name}")
    return args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default="Synthetic",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')
    parser.add_argument('--x_dim', type=int, default=15,
                        help='x dim of synthetic dataset')

    parser.add_argument('--dataset_name', type=str, default='partially_linear_syn',
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
    parser.add_argument('--alpha', type=float, default=0.1,
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

    parser.add_argument('--training_ratio', type=float, default=0.5,
                        help="fraction of samples used for test")
    parser.add_argument('--validation_ratio', type=float, default=0.1,
                        help="fraction of samples used for validation")
    parser.add_argument('--calibration_ratio', type=float, default=0.2,
                        help="fraction of samples used for validation")
    parser.add_argument('--epochs', type=int, default=5000,
                        help="number of epochs for offline training")
    parser.add_argument('--figures_dir', type=str, default='./figures',
                        help="figures_dir")
    parser.add_argument('--saved_models_path', type=str, default='./saved_models',
                        help="saved_models_path")
    parser.add_argument('--batch_norm', type=int, default=0,
                        help="batch norm")
    parser.add_argument('--oracles', type=int, default=0,
                        help="whether to use oracle models")

    args = parser.parse_args()
    args = parse_args_utils(args)
    return args


def get_calibration_schemes_aux(dataset_name, x_dim, y_dim, z_dim, args, data_scaler, scaled_y_max: float,
                                scaled_y_min: float,
                                data_masker: DataCorruptionMasker,
                                hidden_dims, dropout, batch_norm, lr, wd, device,
                                saved_models_path: str, figures_dir: str, seed) -> List[Calibration]:
    alpha = args.alpha
    calibration_schemes = [DummyCalibration(alpha), CQRCalibration(alpha), CQRCalibration(alpha,
                                                                                          ignore_masked=True)]

    def get_data_mask_estimators() -> List[DataMaskEstimator]:
        return [
            get_data_learning_mask_estimator(args, x_dim, z_dim),
            get_data_learning_mask_estimator(args, x_dim, 0),
            OracleDataMasker(data_scaler, data_masker, dataset_name, x_dim, z_dim),
        ]

    for data_mask_estimator in get_data_mask_estimators():
        proxy_qr_model = get_proxy_qr_model(dataset_name, args, x_dim, z_dim, alpha)
        calibration_schemes.append(
            ConservativeWeightedCalibration(CQRCalibration(alpha), CQRCalibration(alpha), alpha, dataset_name,
                                            data_scaler, proxy_qr_model, data_mask_estimator))

    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            WeightedCalibration(CQRCalibration(alpha), alpha, dataset_name, data_scaler, data_mask_estimator))

    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            PrivilegedConformalPrediction(CQRCalibration(alpha), alpha, dataset_name, data_scaler, data_mask_estimator))

    return calibration_schemes


def get_calibration_schemes(dataset_name, x_dim, y_dim, z_dim, args, data_scaler, scaled_y_max: float,
                            scaled_y_min: float,
                            data_masker: DataCorruptionMasker,
                            hidden_dims, dropout, batch_norm, lr, wd, device,
                            saved_models_path: str, figures_dir: str, seed) -> List[Calibration]:
    return get_calibration_schemes_aux(dataset_name, x_dim, y_dim, z_dim, args, data_scaler, scaled_y_max,
                                       scaled_y_min, data_masker,
                                       hidden_dims, dropout, batch_norm, lr, wd, device,
                                       saved_models_path, figures_dir, seed)


def get_model_aux(args, dataset: Dataset) -> PredictionIntervalModel:
    alpha = args.alpha
    if is_data_for_xgboost(dataset.dataset_name):
        model = XGBoostQR(dataset.dataset_name, args.saved_models_path, seed=args.seed, alpha=alpha,
                          )
    else:
        model = QuantileRegression(dataset.dataset_name, args.saved_models_path, dataset.x_dim, dataset.y_dim, alpha,
                                   hidden_dims=args.hidden_dims, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                   device=args.device, figures_dir=args.figures_dir, seed=args.seed)
    return model


def get_models(args, dataset: Dataset) -> List[PredictionIntervalModel]:
    return [get_model_aux(args, dataset)]


def main(args=None):
    if args is None:
        args = parse_args()
    args.z_dim = get_z_dim_from_data_name(args.dataset_name)
    print(f"starting seed: {args.seed} data: {args.dataset_name}")
    set_seeds(args.seed)
    dataset = get_regression_dataset(args)
    set_seeds(args.seed)
    results_helper = RegressionResultsHelper(args.base_results_save_dir, args.seed)
    if 'facebook' in dataset.dataset_name:
        args.hidden_dims = [64, 128, 64, 32]
    else:
        args.hidden_dims = [32, 64, 64, 32]
    models = get_models(args, dataset)
    for model in models:
        model.fit(dataset.x_train, dataset.y_train, dataset.deleted_train, dataset.x_val, dataset.y_val,
                  dataset.deleted_val, args.epochs, args.bs, args.wait)
        model.eval()
        calibration_schemes = get_calibration_schemes(dataset.dataset_name, dataset.x_dim, dataset.y_dim, dataset.z_dim,
                                                      args,
                                                      dataset.scaler,
                                                      dataset.scaled_y_max, dataset.scaled_y_min,
                                                      dataset.data_masker,
                                                      args.hidden_dims, args.dropout, args.batch_norm, args.lr, args.wd,
                                                      args.device,
                                                      args.saved_models_path,
                                                      args.figures_dir,
                                                      args.seed)

        train_uncalibrated_intervals = model.construct_uncalibrated_intervals(dataset.x_train)
        val_uncalibrated_intervals = model.construct_uncalibrated_intervals(dataset.x_val)
        cal_uncalibrated_intervals = model.construct_uncalibrated_intervals(dataset.x_cal)
        test_uncalibrated_intervals = model.construct_uncalibrated_intervals(dataset.x_test)
        for calibration_scheme in calibration_schemes:
            try:
                print(f"started working on calibration: {calibration_scheme.name} with model {model.name}")
                calibration_scheme.fit(dataset.x_train, dataset.y_train, dataset.z_train, dataset.deleted_train,
                                       dataset.x_val,
                                       dataset.y_val, dataset.z_val,
                                       dataset.deleted_val, epochs=args.epochs, batch_size=args.bs, n_wait=args.wait,
                                       train_uncalibrated_intervals=train_uncalibrated_intervals,
                                       val_uncalibrated_intervals=val_uncalibrated_intervals)
                calibration_scheme.calibrate(dataset.x_cal, dataset.y_cal, dataset.z_cal, dataset.deleted_cal,
                                             cal_uncalibrated_intervals)
                test_calibrated_intervals = calibration_scheme.construct_calibrated_uncertainty_sets(dataset.x_test,
                                                                                                     test_uncalibrated_intervals,
                                                                                                     z_test=dataset.z_test)
                train_calibrated_intervals = calibration_scheme.construct_calibrated_uncertainty_sets(dataset.x_train,
                                                                                                      train_uncalibrated_intervals,
                                                                                                      z_test=dataset.z_train)

                results_helper.save_performance_metrics(train_uncalibrated_intervals, train_calibrated_intervals,
                                                        test_uncalibrated_intervals, test_calibrated_intervals,
                                                        dataset, model, calibration_scheme)
            except Exception as e:
                print(f"error: failed on calibration scheme: {calibration_scheme.name} because {e}")
                traceback.print_exc()
                print()


if __name__ == '__main__':
    main()
