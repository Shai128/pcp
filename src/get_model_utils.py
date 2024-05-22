from config.config import datasets_for_xgboost, datasets_for_cnn, datasets_for_rf
from models.data_mask_estimators.DataMaskEstimator import DataMaskEstimator
from models.data_mask_estimators.NetworkMaskEstimator import XGBoostMaskEstimator, NetworkMaskEstimator, \
    CNNMaskEstimator, RFMaskEstimator
from models.qr_models.CNNQuantileRegression import CNNQuantileRegression
from models.qr_models.QuantileRegression import QuantileRegression
from models.qr_models.XGBoostQR import XGBoostQR


def is_data_for_xgboost(dataset_name):
    return any([d in dataset_name for d in datasets_for_xgboost])


def is_data_for_rf(dataset_name):
    return any([d in dataset_name for d in datasets_for_rf])


def is_data_for_cnn(dataset_name):
    return any([d in dataset_name for d in datasets_for_cnn])


def get_data_learning_mask_estimator(args, x_dim, z_dim) -> DataMaskEstimator:
    device = args.device
    if is_data_for_cnn(args.dataset_name) and x_dim > 0:
        return CNNMaskEstimator(args.dataset_name, args.saved_models_path, x_dim, 0,
                                args.hidden_dims, args.dropout, args.batch_norm, args.lr, args.wd, device=args.device,
                                figures_dir=args.figures_dir,
                                seed=args.seed)
    elif is_data_for_xgboost(args.dataset_name):
        return XGBoostMaskEstimator(args.dataset_name, args.saved_models_path, x_dim, z_dim, device=device,
                                    seed=args.seed)
    elif is_data_for_rf(args.dataset_name):
        return RFMaskEstimator(args.dataset_name, args.saved_models_path, x_dim, z_dim, device=device,
                               seed=args.seed)
    else:
        return NetworkMaskEstimator(args.dataset_name, args.saved_models_path, x_dim, z_dim, args.hidden_dims,
                                    args.dropout,
                                    args.batch_norm, args.lr, args.wd, device=device, figures_dir=args.figures_dir,
                                    seed=args.seed)


def get_proxy_qr_model(dataset_name, args, x_dim, z_dim, alpha, **kwargs):
    if is_data_for_xgboost(dataset_name):
        return XGBoostQR(dataset_name + '_z', args.saved_models_path, args.seed, alpha)
    elif is_data_for_cnn(dataset_name):
        return CNNQuantileRegression(dataset_name + '_z', args.saved_models_path, x_dim, z_dim, alpha,
                                  hidden_dims=args.hidden_dims, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                  device=args.device, figures_dir=args.figures_dir, seed=args.seed)
    else:
        return QuantileRegression(dataset_name + '_z', args.saved_models_path, x_dim, z_dim, alpha,
                                  hidden_dims=args.hidden_dims, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                  device=args.device, figures_dir=args.figures_dir, seed=args.seed)
