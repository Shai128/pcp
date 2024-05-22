import argparse
import ast
import traceback
import warnings
from typing import List

import torch

from calibration_schemes.APSCalibration import APSCalibration
from calibration_schemes.AbstractCalibration import Calibration
from calibration_schemes.CQRCalibration import CQRCalibration
from calibration_schemes.TwoStagedConformalPrediction import ConservativeWeightedCalibration
from calibration_schemes.HPSCalibration import HPSCalibration
from calibration_schemes.PrivilegedConformalPrediction import PrivilegedConformalPrediction
from calibration_schemes.OracleAPSCalibration import OracleAPSCalibration
from calibration_schemes.OracleHPSCalibration import OracleHPSCalibration
from calibration_schemes.WeightedCalibration import WeightedCalibration
from data_utils.data_corruption.data_corruption_masker import DataCorruptionMasker, DummyDataCorruptionMasker
from data_utils.data_type import DataType
from data_utils.datasets.classification_dataset import ClassificationDataset
from data_utils.get_dataset_utils import get_classification_dataset
from get_model_utils import get_data_learning_mask_estimator, get_proxy_qr_model, is_data_for_xgboost, is_data_for_cnn, \
    is_data_for_rf
from models.classifiers.NetworkClassifier import NetworkClassifier
from models.classifiers.RFClassifier import RFClassifier
from models.classifiers.XGBClassifier import XGBClassifier
from models.classifiers.CNNClassifier import CNNClassifier
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.data_mask_estimators.OracleDataMasker import OracleDataMasker
from models.ClassificationModel import ClassificationModel
from results_helper.classification_results_helper import ClassificationResultsHelper
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
    parser.add_argument('--data_type', type=str, default="Synthetic",
                        help='type of data set. real or synthetic. REAL for real. SYN for synthetic')
    parser.add_argument('--x_dim', type=int, default=10,
                        help='x dim of synthetic dataset')
    parser.add_argument('--n_classes', type=int, default=10,
                        help='n_classes synthetic dataset')

    parser.add_argument('--dataset_name', type=str, default='classification_synthetic',
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
    parser.add_argument('--wait', type=int, default=700,
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help="number of epochs for offline training")
    parser.add_argument('--figures_dir', type=str, default='./figures',
                        help="figures_dir")
    parser.add_argument('--saved_models_path', type=str, default='./saved_models',
                        help="saved_models_path")
    parser.add_argument('--batch_norm', type=int, default=0,
                        help="batch norm")

    args = parser.parse_args()
    args = parse_args_utils(args)
    return args



def get_calibration_schemes(args, dataset_name, x_dim, z_dim, n_classes, data_scaler,
                            data_masker: DataCorruptionMasker) -> List[Calibration]:
    alpha = args.alpha
    use_x = not is_data_for_cnn(dataset_name)
    # return [APSCalibration(alpha), APSCalibration(alpha, ignore_masked=True), OracleAPSCalibration(alpha)]
    calibration_schemes = [HPSCalibration(alpha), HPSCalibration(alpha, ignore_masked=True),
                           APSCalibration(alpha), APSCalibration(alpha, ignore_masked=True),
                           OracleHPSCalibration(alpha), OracleAPSCalibration(alpha)]

    def get_data_mask_estimators() -> List[DataMaskEstimator]:
        curr_x_dim = x_dim if use_x else 0
        estimators: List[DataMaskEstimator] = [
            get_data_learning_mask_estimator(args, curr_x_dim, z_dim),
            get_data_learning_mask_estimator(args, x_dim, 0),
        ]
        if not isinstance(data_masker, DummyDataCorruptionMasker):
            estimators = [OracleDataMasker(data_scaler, data_masker, dataset_name, x_dim, z_dim),
                          *estimators]
        return estimators

    try:
        for data_mask_estimator in get_data_mask_estimators():
            proxy_qr_model = get_proxy_qr_model(dataset_name, args, x_dim, z_dim, alpha)
            calibration_schemes.append(
                ConservativeWeightedCalibration(CQRCalibration(alpha), HPSCalibration(alpha), alpha, dataset_name,
                                                data_scaler, proxy_qr_model, data_mask_estimator))
    except Exception as e:
        traceback.print_exc()
        print(f"failed creating two-staged calibration because: {e}")

    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            WeightedCalibration(HPSCalibration(alpha), alpha, dataset_name, data_scaler, data_mask_estimator))

    for data_mask_estimator in get_data_mask_estimators():
        calibration_schemes.append(
            PrivilegedConformalPrediction(HPSCalibration(alpha), alpha, dataset_name, data_scaler, data_mask_estimator))

    return calibration_schemes


def get_model(args, dataset: ClassificationDataset) -> ClassificationModel:
    if is_data_for_cnn(dataset.dataset_name):
        model = CNNClassifier(dataset.dataset_name, args.saved_models_path, dataset.x_dim, z_dim=0,
                              n_classes=dataset.n_classes,
                              hidden_dims=args.hidden_dims, dropout=args.dropout, lr=args.lr, wd=args.wd,
                              device=args.device, figures_dir=args.figures_dir, seed=args.seed)
    elif is_data_for_xgboost(dataset.dataset_name):
        model = XGBClassifier(dataset.dataset_name, args.saved_models_path, args.x_dim, dataset.n_classes, args.device,
                              seed=args.seed)
    elif is_data_for_rf(dataset.dataset_name):
        model = RFClassifier(dataset.dataset_name, args.saved_models_path, args.x_dim, dataset.n_classes, args.device,
                             seed=args.seed)
    else:
        model = NetworkClassifier(dataset.dataset_name, args.saved_models_path, dataset.x_dim, dataset.n_classes,
                                  args.hidden_dims, args.dropout,
                                  lr=args.lr, wd=args.wd, device=args.device, figures_dir=args.figures_dir,
                                  seed=args.seed)
    return model


def main(args):
    print(f"starting seed: {args.seed} data: {args.dataset_name}")
    set_seeds(args.seed)
    dataset = get_classification_dataset(args)
    model = get_model(args, dataset)
    calibration_schemes = get_calibration_schemes(args, dataset.dataset_name, dataset.x_dim,
                                                  dataset.z_dim, dataset.n_classes, dataset.scaler,
                                                  dataset.data_masker)

    model.fit(dataset.x_train, dataset.y_train, dataset.deleted_train, dataset.x_val, dataset.y_val,
              dataset.deleted_val, args.epochs, args.bs, args.wait)
    model.eval()

    results_helper = ClassificationResultsHelper(args.base_results_save_dir, args.seed)
    train_model_prediction = model.estimate_probabilities(dataset.x_train)
    cal_model_prediction = model.estimate_probabilities(dataset.x_cal)
    test_proba_prediction = model.estimate_probabilities(dataset.x_test)
    probability_sum = test_proba_prediction.probabilities.sum(dim=-1)
    assert ((probability_sum >= 0 - 0.01) & (probability_sum <= 1 + 0.01)).all()
    for calibration_scheme in calibration_schemes:
        try:
            calibration_scheme.fit(dataset.x_train, dataset.y_train, dataset.z_train, dataset.deleted_train,
                                   dataset.x_val,
                                   dataset.y_val, dataset.z_val,
                                   dataset.deleted_val, epochs=args.epochs, batch_size=args.bs, n_wait=args.wait)
            calibration_scheme.calibrate(dataset.x_cal, dataset.y_cal, dataset.z_cal, dataset.deleted_cal,
                                         cal_model_prediction,
                                         full_y_cal=dataset.full_y_cal)
            test_calibrated_intervals = calibration_scheme.construct_calibrated_uncertainty_sets(dataset.x_test,
                                                                                                 test_proba_prediction,
                                                                                                 z_test=dataset.z_test)

            train_calibrated_intervals = calibration_scheme.construct_calibrated_uncertainty_sets(dataset.x_train,
                                                                                                  train_model_prediction,
                                                                                                  z_test=dataset.z_train)
            results_helper.save_performance_metrics(train_model_prediction, train_calibrated_intervals,
                                                    test_proba_prediction, test_calibrated_intervals,
                                                    dataset, model, calibration_scheme)
        except Exception as e:
            print(f"error: failed on calibration scheme: {calibration_scheme.name} because {e}")
            traceback.print_exc()
            print()


if __name__ == '__main__':
    args = parse_args()
    main(args)
