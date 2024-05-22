import os
import re
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import scipy
import torch
import torchvision
from robustbench.data import load_cifar10c
import matplotlib.pyplot as plt
import utils
from data_utils.data_corruption.corruption_type import remove_corruption_type_from_dataset_name, \
    get_corruption_type_from_dataset_name
from data_utils.data_corruption.covariates_dimension_reducer import ZMeanReducer, MeanReducer, \
    CovariatesDimensionReducer
from data_utils.data_corruption.data_corruption_masker import DataCorruptionIndicatorFactory, DummyDataCorruptionMasker, \
    OracleDataCorruptionMasker, DefaultDataCorruptionMasker
from data_utils.data_scaler import DataScaler
from data_utils.data_type import DataType
from data_utils.datasets.classification_dataset import ClassificationDataset
from data_utils.datasets.regression_dataset import RegressionDataset
from data_utils.datasets.synthetic_causal_inference_data_generator import CausalInferenceDataGenerator
from data_utils.datasets.synthetic_dataset_generator import PartiallyLinearDataGenerator, SyntheticDataGenerator
from models.LinearModel import LinearModel, LogisticLinearModel
from models.classifiers.NetworkClassifier import NetworkClassifier
from models.classifiers.XGBClassifier import XGBClassifier
from models.regressors.FullRegressor import FullRegressor
from utils import set_seeds

proxy_col_dict = {
    'meps_19': [3],
    'meps_20': [3],
    'meps_21': [3],
    'facebook_1': [33, 11],
    'facebook_2': [33, 11],
    'bio': [2, 3],
    'house': [2, 8],  # 14,
    "blog": [9, 20]
}


def remove_z_dim_from_dataset_name(dataset_name: str) -> str:
    z_dim_part = re.search(r'_z\d+', dataset_name)
    if z_dim_part is None:
        return dataset_name
    return dataset_name.replace(re.search(r'_z\d+', dataset_name).group(), "")


def get_original_dataset_name(dataset_name: str) -> str:
    dataset_name = remove_z_dim_from_dataset_name(dataset_name)
    dataset_name = remove_corruption_type_from_dataset_name(dataset_name)
    return dataset_name


def get_z_dim_from_data_name(dataset_name: str):
    z_dim_part = re.search(r'_z\d+', dataset_name)
    if z_dim_part is None:
        z_dim = 1
    else:
        z_dim = int(z_dim_part.group().replace("_z", ""))
    return z_dim


def get_noised_ihdp_data(args):
    x, y, z, corruption_masker, mask = get_ihdp_data(args)
    set_seeds(args.seed)
    model_x = torch.cat([x, z], dim=-1).to(args.device)
    model_y = mask.to(args.device)
    classifier = XGBClassifier(args.dataset_name + '_y(0)', args.saved_models_path, model_x.shape[1], 2,
                               device=args.device, max_depth=2, n_estimators=10, seed=args.seed)
    permutation = np.random.permutation(len(model_x))
    train_idx = permutation[:int(len(permutation) * 0.8)]
    val_idx = permutation[int(len(permutation) * 0.8):]
    classifier.fit_xy(model_x[train_idx], model_y[train_idx], None, model_x[val_idx], model_y[val_idx], None,
                      epochs=1000, batch_size=args.bs, n_wait=args.wait)
    propensity = classifier.estimate_probabilities(model_x).probabilities.detach()[..., 1].cpu()
    correction = mask.float().mean() - propensity.mean()
    propensity += correction
    high_mask_probability_idx = propensity >= torch.quantile(propensity, q=0.8)

    y_std = y.std()
    noise = torch.randn_like(y) * (y_std * 5)
    y[high_mask_probability_idx] += noise[high_mask_probability_idx]
    set_seeds(args.seed)
    return x, y, z, corruption_masker, mask


def get_ihdp_data(args):
    # Get the data from here: https://www.fredjo.com/
    # Was used in "Estimating individual treatment effect: generalization bounds and algorithms"
    # Was used also in "Causal Effect Inference with Deep Latent-Variable Models"
    # TODO: find more related papers
    # TODO: maybe artificially ruin the performance of the naive method
    data1 = np.load(os.path.join(args.data_path, "ihdp_npci_1-1000.train.npz"))
    data2 = np.load(os.path.join(args.data_path, "ihdp_npci_1-1000.test.npz"))
    x, t, yf, ycf = [], [], [], []
    for data in [data1, data2]:
        x += [torch.Tensor(data.get("x"))]
        t += [torch.Tensor(data.get("t"))]
        yf += [torch.Tensor(data.get("yf"))]
        ycf += [torch.Tensor(data.get("ycf"))]
    x = torch.cat(x, dim=0)
    t = torch.cat(t, dim=0)
    yf = torch.cat(yf, dim=0)
    ycf = torch.cat(ycf, dim=0)
    t = t[..., args.seed].bool()
    x = x[..., args.seed]
    yf = yf[..., args.seed]
    ycf = ycf[..., args.seed]
    y = yf.clone()
    y[t] = ycf[t].clone()

    # [np.corrcoef(mask.float().numpy(), x.numpy()[:, i]) for i in range(x.shape[1])]
    # relevant_cols = [i for i in range(x.shape[1]) if len(torch.unique(x[:, i])) > 5]
    relevant_cols = range(x.shape[1])
    proxy_col = np.argmax([abs(np.corrcoef(y.float().numpy(), x.numpy()[:, i])[0, 1]) for i in relevant_cols])
    assert proxy_col == 0
    z = x[:, proxy_col]
    if len(z.shape) == 1:
        z = z.unsqueeze(-1)
    cols_mask = np.ones(x.shape[1], dtype=bool)
    cols_mask[proxy_col] = False
    x = x[:, cols_mask]

    model = LinearModel("ihdp_x_y_model", args.saved_models_path, args.figures_dir, args.seed)
    model.fit(torch.cat([x, z], dim=-1), y)

    class LinearModelReducer(CovariatesDimensionReducer):

        def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if len(z.shape) == 1:
                z = z.unsqueeze(-1)
            return model.predict(torch.cat([x, z], dim=-1)).squeeze()

    corruption_masker = DefaultDataCorruptionMasker('ihdp', LinearModelReducer(),
                                                    unscaled_full_x=x,
                                                    unscaled_full_z=z,
                                                    marginal_masking_ratio=0.2
                                                    )
    mask = corruption_masker.get_corruption_mask(unscaled_x=x, unscaled_z=z, seed=args.seed).bool()

    return x, y, z, corruption_masker, mask


def get_nlsm_data(args):
    # -1: Was used in https://arxiv.org/pdf/2006.06138.pdf (Conformal Inference
    # of Counterfactuals and Individual Treatment Effects)
    # 0: Raw data was taken from: https://github.com/grf-labs/grf/blob/master/experiments/acic18/synthetic_data.csv
    # 1: real trial: Assessing treatment effect variation in observational studies: Results from a data challenge
    # 2: synthetic data based on it: A national experiment reveals where a growth mindset improves achievement
    # 3: The artificial definition of the data is based on "Sensitivity Analysis of Individual Treatment Effects:
    # A Robust Conformal Inference Approach" which is based on "Assessing treatment effect variation
    # in observational studies: Results from a data challenge"

    set_seeds(0)
    df = pd.read_csv(os.path.join(args.data_path, 'nslm.csv'))
    mask = torch.Tensor(df['Z'].values).bool().to(args.device)
    y = torch.Tensor(df['Y'].values).to(args.device)
    X_cols = ['S3', 'C1', 'C2', 'C3', 'XC', 'X1', 'X2', 'X3', 'X4', 'X5']
    x1_col = X_cols.index("X1")
    x2_col = X_cols.index("X2")
    c1_col = X_cols.index("C1")
    x = torch.Tensor(df[X_cols].values).to(args.device)
    temp_scaler = DataScaler()
    temp_scaler.initialize_scalers(x, y)
    x = temp_scaler.scale_x(x)
    y = temp_scaler.scale_y(y)

    regressor = FullRegressor(args.dataset_name + "_y(0)", args.saved_models_path, x.shape[1], 0,
                              [32],
                              args.dropout, False, args.lr, args.wd, args.device,
                              figures_dir=args.figures_dir,
                              seed=0)
    permutation = np.random.permutation(len(x[~mask]))
    train_idx = permutation[:int(len(permutation) * 0.8)]
    val_idx = permutation[int(len(permutation) * 0.8):]
    regressor.fit(x[~mask][train_idx], y[~mask][train_idx], None, None,
                  x[~mask][val_idx], y[~mask][val_idx], None, None, epochs=1000, batch_size=args.bs, n_wait=args.wait)
    #
    # classifier = NetworkClassifier(args.dataset_name + "_y(0)", args.saved_models_path, x.shape[1], 2,
    #                           [32],
    #                           args.dropout, False, args.lr, args.wd, args.device,
    #                           figures_dir=args.figures_dir,
    #                           seed=0)
    classifier = XGBClassifier(args.dataset_name + '_y(0)', args.saved_models_path, x.shape[1], 2, device=args.device,
                               max_depth=2, n_estimators=10, seed=0)
    permutation = np.random.permutation(len(x))
    train_idx = permutation[:int(len(permutation) * 0.8)]
    val_idx = permutation[int(len(permutation) * 0.8):]
    classifier.fit_xy(x[train_idx], mask[train_idx], None, x[val_idx], mask[val_idx], None, epochs=1000,
                      batch_size=args.bs, n_wait=args.wait)
    propensity = classifier.estimate_probabilities(x).probabilities.detach()[..., 1]
    correction = mask.float().mean() - propensity.mean()
    propensity += correction

    mu_hat_0 = temp_scaler.unscale_y(regressor.predict_mean(x, None).detach())
    x = temp_scaler.unscale_x(x)
    y = temp_scaler.unscale_y(y)

    u = torch.randn_like(y) * 0.2
    is_u_extreme = (u >= torch.quantile(u, q=0.9)) | (u <= torch.quantile(u, q=0.1))
    # new_propensity = propensity
    new_propensity = propensity * 2 * is_u_extreme + propensity * (~is_u_extreme)
    new_propensity = torch.min(new_propensity, 0.8 * torch.ones_like(new_propensity))

    # mask = torch.rand_like(new_propensity) < new_propensity

    c1_is_in_set = (abs(x[:, c1_col] - 1) < 0.1) | (abs(x[:, c1_col] - 13) < 0.1) | (abs(x[:, c1_col] - 14) < 0.1)
    tau = 0.228 + 0.05 * (x[:, x1_col] < 0.07) - 0.05 * (x[:, x2_col] < -0.69) - 0.08 * c1_is_in_set
    new_u = 2 * u * is_u_extreme + u * (~ is_u_extreme)
    new_y = mu_hat_0 + tau + new_u

    z = u.unsqueeze(-1)

    corruption_masker = OracleDataCorruptionMasker(x, z, new_propensity)
    new_mask = corruption_masker.get_corruption_mask(x, z)

    # corruption_probabilities = corruption_masker.get_corruption_probabilities(x, z)
    # err = (corruption_probabilities - new_propensity).abs().max().item()
    # if err > 0:
    #     print(f"probability error: err={err}")

    set_seeds(args.seed)

    return x.cpu(), new_y.cpu(), z.cpu(), corruption_masker, new_mask.cpu()


def get_twins_data(args):
    # taken from https://github.com/py-why/dowhy/blob/main/docs/source/example_notebooks/dowhy_twins_example.ipynb
    x_df = pd.read_csv(os.path.join(args.data_path, "twins", 'twins_x.csv'))
    y_df = pd.read_csv(os.path.join(args.data_path, "twins", 'twins_y.csv'))[["mort_0", "mort_1"]]
    t_df = pd.read_csv(os.path.join(args.data_path, "twins", 'twins_t.csv'))[["dbirwt_0", "dbirwt_1"]]
    lighter_features = x_df[['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
                             'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
                             'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
                             'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
                             'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
                             'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
                             'data_year', 'nprevistq', 'dfageq', 'feduc6', 'infant_id_0',
                             'dlivord_min', 'dtotord_min', 'bord_0',
                             'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']]
    lighter_features.fillna(value=lighter_features.mean(axis="rows"), inplace=True)
    # lighter_features["dbirwt_0"] = t["dbirwt_0"]
    relevant_idx = (t_df <= 2000).all(axis='columns')
    x = torch.Tensor(lighter_features.to_numpy())[relevant_idx]
    z = torch.Tensor(t_df["dbirwt_0"].to_numpy()).unsqueeze(-1)[relevant_idx]
    y = torch.Tensor(y_df["mort_0"].to_numpy()).unsqueeze(-1)[relevant_idx]
    """
    t2 = t.copy().to_numpy()
    t2[:, 0] = 0
    t2[:, 1] = 1
    y.to_numpy().flatten()
    scipy.stats.pearsonr(t2.flatten(), y.to_numpy().flatten())
    """

    model = LogisticLinearModel("twins_x_y_model", args.saved_models_path, args.figures_dir, args.seed)
    model.fit(torch.cat([x, z], dim=-1), y)

    class LinearModelReducer(CovariatesDimensionReducer):

        def __call__(self, x: torch.Tensor, z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if len(z.shape) == 1:
                z = z.unsqueeze(-1)
            return model.estimate_probabilities(torch.cat([x, z], dim=-1)).squeeze()[:, 1]

    corruption_masker = DefaultDataCorruptionMasker('twins', LinearModelReducer(),
                                                    unscaled_full_x=x,
                                                    unscaled_full_z=z,
                                                    marginal_masking_ratio=0.2)

    mask = corruption_masker.get_corruption_mask(unscaled_x=x, unscaled_z=z, seed=args.seed).bool()

    return x, None, y, z, corruption_masker, mask


def get_real_data(args):
    # some of the datasets are taken from: https://github.com/py-why/dowhy/blob/main/dowhy/datasets.py
    dataset_name = args.dataset_name
    if 'noised_ihdp' in dataset_name:
        return get_noised_ihdp_data(args)
    elif 'ihdp' in dataset_name:
        return get_ihdp_data(args)
    elif 'nslm' in dataset_name:
        return get_nlsm_data(args)
    z_dim = args.z_dim
    original_data_name = get_original_dataset_name(dataset_name)
    corruption_type = get_corruption_type_from_dataset_name(dataset_name)
    X, y = load_real_regression_dataset(original_data_name, args.data_path)
    X = torch.Tensor(X)
    y = torch.Tensor(y)
    # [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
    # np.argsort([abs(np.corrcoef(X[:, i], y)[0,1]) for i in range(X.shape[1])])
    # [j for j in np.argsort([abs(np.corrcoef(X[:, i], y)[0,1]) for i in range(X.shape[1])]) if len(X[:, j].unique()) > 10]
    if len(proxy_col_dict[original_data_name]) < z_dim:
        raise Exception("too few proxy columns")
    proxy_cols = proxy_col_dict[original_data_name][:z_dim]
    # if isinstance(proxy_col, list):
    #     proxy_col = proxy_col[0]
    # if dataset_name == 'facebook_1' or  dataset_name == 'facebook_2':
    #     X[:, response_col] = torch.log(X[:, response_col] - X[:, response_col].min() + 1)
    #     y = torch.log(y - y.min() + 1)
    # if dataset_name == 'popularity':
    #     y = torch.log(y - y.min() + 1)

    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('module://backend_interagg')
    # response_col = 3
    # plt.hist(torch.log(X[:, response_col] - X[:, response_col].min() + 1).numpy(), bins=20)
    # plt.xlabel("log x")
    # plt.ylabel("count")
    # plt.show()
    # plt.hist(X[:, response_col].numpy(), bins=20)
    # plt.xlabel("x")
    # plt.ylabel("count")
    # plt.show()
    # plt.hist(y.numpy(), bins=20)
    # plt.xlabel("y")
    # plt.ylabel("count")
    # plt.show()
    # plt.hist(torch.log(y - y.min() + 1).numpy(), bins=20)
    # plt.xlabel("log(y)")
    # plt.ylabel("count")
    # plt.show()
    # plt.scatter(X[:, response_col], y)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    Z = X[:, proxy_cols]
    # if z_dim > 1:
    #     more_z = generate_proxy_variable(y.squeeze(), z_dim - 1)
    #     Z = torch.cat([Z.unsqueeze(-1), more_z], dim=-1)

    Y = y.unsqueeze(1)
    # Y[:, 0] = torch.log(Y[:, 0] - Y[:,0].min() + 1)
    # if args.figures_dir is not None:
    #     plt.clf()
    #     plt.scatter(Y[:, 0], Y[:, 1])
    #     save_dir = os.path.join(args.figures_dir, dataset_name, f'seed={args.seed}')
    #     create_folder_if_it_doesnt_exist(save_dir)
    #     save_path = os.path.join(save_dir, "data_visualization.png")
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.show()

    mask = np.ones(X.shape[1], dtype=bool)
    mask[proxy_cols[0]] = False
    X = X[:, mask]
    corruption_masker = DataCorruptionIndicatorFactory.get_corruption_masker(dataset_name, corruption_type, X, Z,
                                                                             Y)

    mask = corruption_masker.get_corruption_mask(X, Z)
    return X, Y, Z, corruption_masker, mask


def get_data_generator(dataset_name, x_dim, z_dim) -> SyntheticDataGenerator:
    original_dataset_name = get_original_dataset_name(dataset_name)
    corruption_type = get_corruption_type_from_dataset_name(dataset_name)
    if original_dataset_name == 'regression_synthetic':
        return PartiallyLinearDataGenerator(x_dim, z_dim, corruption_type)
    if original_dataset_name == 'synthetic_causal':
        return CausalInferenceDataGenerator(x_dim)
    else:
        raise Exception(f"unknown dataset name: {dataset_name} and cannot construct a generator for it")


def get_regression_dataset(args) -> RegressionDataset:
    dataset_name: str = args.dataset_name
    if args.data_type == DataType.Real:
        x, y, z, data_masker, d = get_real_data(args)
    else:
        data_generator = get_data_generator(dataset_name, args.x_dim, args.z_dim)
        x, y, z, data_masker, d = data_generator.generate_data(args.data_size, args.device)
    if len(d.shape) == 2:
        marginal_mask_probability = d.any(dim=1).float().mean().item()
    elif len(d.shape) == 1:
        marginal_mask_probability = d.float().mean().item()
    else:
        raise Exception(f"don't know how to handle with len(d.shape)={len(d.shape)}")
    if marginal_mask_probability > 0.5 or marginal_mask_probability < 0.05:
        print(f"warning: marginal mask ratio={marginal_mask_probability}")
    dataset = RegressionDataset(x, y, z, d, data_masker, dataset_name, args.training_ratio, args.validation_ratio,
                                args.calibration_ratio, args.device,
                                args.saved_models_path, args.figures_dir, args.seed)
    print(f"data size: {x.shape[0]}, x_dim: {dataset.x_dim}, y_dim: {dataset.y_dim} z_dim: {dataset.z_dim}")
    return dataset


def load_real_regression_dataset(name: str, base_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Load a dataset
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/RCOL_datasets/directory/"
    Returns
    -------
    X : features (nXp)
    y : labels (n)
	"""

    if name == "meps_19":
        df = pd.read_csv(os.path.join(base_path, 'meps_19_reg_fix.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    elif name == "meps_20":
        df = pd.read_csv(os.path.join(base_path, 'facebook/meps_20_reg_fix.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    elif name == "meps_21":

        df = pd.read_csv(os.path.join(base_path, 'facebook/meps_21_reg_fix.csv'))
        column_names = df.columns
        response_name = "UTILIZATION_reg"
        column_names = column_names[column_names != response_name]
        column_names = column_names[column_names != "Unnamed: 0"]

        col_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT16F', 'REGION=1',
                     'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                     'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                     'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                     'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                     'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                     'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                     'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                     'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                     'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                     'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                     'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                     'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                     'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                     'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                     'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                     'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                     'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                     'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                     'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                     'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                     'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                     'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                     'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                     'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                     'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                     'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        y = df[response_name].values
        X = df[col_names].values

    elif name == "facebook_1":

        df = pd.read_csv(os.path.join(base_path, 'facebook/Features_Variant_1.csv'))
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    elif name == "facebook_2":
        df = pd.read_csv(os.path.join(base_path, 'facebook/Features_Variant_2.csv'))
        y = df.iloc[:, 53].values
        X = df.iloc[:, 0:53].values

    elif name == "bio":
        # https://github.com/joefavergel/TertiaryPhysicochemicalProperties/blob/master/RMSD-ProteinTertiaryStructures.ipynb
        df = pd.read_csv(os.path.join(base_path, 'CASP.csv'))
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    elif name == 'house':
        df = pd.read_csv(os.path.join(base_path, 'kc_house_data.csv'))
        y = np.array(df['price'])
        X = (df.drop(['id', 'date', 'price'], axis=1)).values
    elif name == 'blog_data' or name == 'blog':
        # https://github.com/xinbinhuang/feature-selection_blogfeedback
        df = pd.read_csv(os.path.join(base_path, 'blogData_train.csv'), header=None)
        X = df.iloc[:, 0:280].values
        y = df.iloc[:, -1].values
    else:
        raise Exception(f"invalid dataset: {name}")

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    return X, y


def get_real_classification_data(args):
    dataset_name = args.dataset_name
    if 'cifar10c' in dataset_name:
        return get_cifar10c_dataset(args)
    if 'cifar10' in dataset_name:
        return get_cifar10_dataset(args)
    elif 'twins' in dataset_name:
        return get_twins_data(args)
    else:
        raise Exception(f"does not know how to handle with data: {dataset_name}")


# corruption_to_flip_probability = {
#     'snow': 0.95,
#     'gaussian_noise': 0.93,
#     'defocus_blur': 0.92,
#     'motion_blur': 0.92,
#     'glass_blur': 0.91,
#     'pixelate': 0.945,
#     'brightness': 0.925,
#     'shot_noise': 0.91,
#     'elastic_transform': 0.87,
#     'contrast': 0.85,
#     'fog': 0.95,
#     'frost': 0.94,
#     'jpeg_compression': 0.9,
#     'impulse_noise': 0.88,
#     'zoom_blur': 0.92,
# }


corruption_to_flip_probability = {
    'snow': 0.95,
    # 'gaussian_noise': 0.93,
    'defocus_blur': 0.9,
    'pixelate': 0.93,
    'fog': 0.94,
    # 'motion_blur': 0.92,
    # 'glass_blur': 0.91,
    # 'brightness': 0.925,
    # 'shot_noise': 0.91,
    # 'elastic_transform': 0.87,
    # 'contrast': 0.85,
    # 'frost': 0.94,
    # 'jpeg_compression': 0.9,
    # 'impulse_noise': 0.88,
    # 'zoom_blur': 0.92,
}


def get_cifar10c_dataset(args):
    if 'adversarial' in args.dataset_name:
        print("cifar10c - using adversarial mode")
        noise_function = adversarial_flip
        min_severity = 4
        # corruption_to_flip_probability = {
        #     'snow': 0.95,
        #     'defocus_blur': 0.92,
        #     'motion_blur': 0.92,
        #     'pixelate': 0.945,
        #     'brightness': 0.925,
        # }

    else:
        min_severity = 4
        noise_function = random_flip

    set_seeds(0)
    clean_data_path = os.path.join(args.data_path, 'cifar10')
    clean_dataset = torchvision.datasets.CIFAR10(root=clean_data_path, train=False, download=True)
    clean_x = torch.Tensor(clean_dataset.data).to(torch.uint8)
    n_labels = len(clean_dataset.classes)
    clean_y = torch.Tensor(clean_dataset.targets)
    overall_data_size = len(clean_x)
    noised_data_path = os.path.join(args.data_path, 'cifar10c')

    severity_to_flip_probability = {
        1: 0.6,
        2: 0.8,
        3: 0.85,
        4: 0.92,
        5: 0.95
    }
    corruptions = list(corruption_to_flip_probability.keys())

    corruption_and_severity_to_x_test: Dict[str, torch.Tensor] = {}
    corruption_and_severity_to_flip_probability: Dict = {}
    for corruption in corruptions:
        for severity in range(min_severity, 5 + 1):
            x_test, y_test = load_cifar10c(10000,
                                           severity, noised_data_path, False,
                                           [corruption])
            x_test = x_test * 255
            x_test = x_test.to(torch.uint8)
            corruption_and_severity_to_x_test[f"{corruption}_{severity}"] = x_test
            severity_flip_probability = severity_to_flip_probability[severity]
            flip_probability = severity_flip_probability * corruption_to_flip_probability[corruption]
            corruption_and_severity_to_flip_probability[f"{corruption}_{severity}"] = flip_probability

    corrupted_data_size = int(4*0.15 * overall_data_size)
    corrupted_indexes = np.random.permutation(overall_data_size)[:corrupted_data_size]
    corruptions_and_severities = np.random.choice(len(corruption_and_severity_to_x_test), size=(corrupted_data_size,),
                                                  replace=True)
    new_x = clean_x.clone()
    new_y = clean_y.clone()
    z = torch.zeros(len(clean_y), 2)
    keys = list(corruption_and_severity_to_x_test.keys())
    flip_probabilities = torch.zeros(len(new_x)).to(new_x.device)
    for idx, corruption_and_severity_idx in zip(corrupted_indexes, corruptions_and_severities):
        new_x[idx] = corruption_and_severity_to_x_test[keys[corruption_and_severity_idx]][idx].transpose(0,
                                                                                                         2).transpose(0,
                                                                                                                      1)
        flip_probability = corruption_and_severity_to_flip_probability[keys[corruption_and_severity_idx]]
        # flip_probability = 0.12
        flip_probabilities[idx] = flip_probability
        severity = int(keys[corruption_and_severity_idx][-1])
        corruption = keys[corruption_and_severity_idx][:-2]
        if np.random.rand(1) < flip_probability:
            new_y[idx] = noise_function(new_y[idx].item(), n_labels, severity, corruption,
                                        corruption_to_flip_probability)

        z[idx, 0] = severity + 1
        z[idx, 1] = corruptions.index(corruption) + 1

    # if 'adversarial' in args.dataset_name:
    train_data_size = 30000
    train_clean_dataset = torchvision.datasets.CIFAR10(root=clean_data_path, train=True, download=True)
    train_clean_x = torch.Tensor(train_clean_dataset.data).to(torch.uint8)[:train_data_size].to(new_x.device)
    train_clean_y = torch.Tensor(train_clean_dataset.targets)[:train_data_size].to(new_x.device)
    new_x = torch.cat([new_x, train_clean_x])
    new_y = torch.cat([new_y, train_clean_y])
    clean_y = torch.cat([clean_y, train_clean_y])
    z = torch.cat([z, torch.zeros(train_data_size, z.shape[1])])
    flip_probabilities = torch.cat([flip_probabilities, torch.zeros(len(train_clean_y)).to(new_x.device)])

    d = new_y != clean_y
    data_masker = OracleDataCorruptionMasker(new_x, z, flip_probabilities)
    idx = np.random.permutation(len(new_x))
    set_seeds(args.seed)
    return new_x[idx], clean_y[idx], new_y[idx], z[idx], data_masker, d[idx]


def random_flip(curr_label: int, n_labels: int, severity: int, corruption: str,
                corruption_to_flip_probability: Dict[str, float]) -> int:
    new_label = np.random.randint(0, n_labels - 1)
    if new_label >= curr_label:
        new_label += 1
    assert new_label != curr_label
    return new_label


def adversarial_flip(curr_label: int, n_labels: int, severity: int, corruption: str,
                     corruption_to_flip_probability: Dict[str, float]) -> int:
    # if np.random.rand(1) < 0.3:
    #     return random_flip(curr_label, n_labels, severity, corruption, corruption_to_flip_probability)
    mid = len(corruption_to_flip_probability.keys()) // 2
    if corruption in list(corruption_to_flip_probability.keys())[:mid]:
        new_label = n_labels // 4
    else:
        new_label = 3 * n_labels // 4
    if new_label == curr_label:
        new_label = n_labels - curr_label - 1
    assert 0 <= new_label <= n_labels - 1
    assert new_label != curr_label
    return new_label


def get_cifar10_dataset(args):
    clean_data_path = os.path.join(args.data_path, 'cifar10')
    data = torch.load(os.path.join(args.data_path, 'CIFAR-10_human.pt'))
    metadata = pd.read_csv(os.path.join(args.data_path, "side_info_cifar10N.csv"))
    noised_labels = torch.Tensor(data['random_label1'])
    noised_labels2 = torch.Tensor(data['random_label2'])
    noised_labels3 = torch.Tensor(data['random_label3'])
    n_unique_labels = []
    for i in range(len(noised_labels)):
        n_unique_labels += [
            len(np.unique([noised_labels[i].item(), noised_labels2[i].item(), noised_labels3[i].item()]))]
    n_unique_labels = torch.Tensor(n_unique_labels)
    clean_labels = torch.Tensor(data['clean_label'])
    n_labels = clean_labels.max().int().item() + 1
    dataset = torchvision.datasets.CIFAR10(root=clean_data_path, train=True, download=True)
    # error = noised_labels != clean_labels

    x = torch.Tensor(dataset.data)
    # x = x.to(torch.uint8)
    # x = x.transpose(1,-1).transpose(2,3)

    # z = metadata['Work-time-in-seconds-1']
    work_times = torch.zeros(len(x), 3)
    for i in range(len(metadata)):
        image_batch = metadata['Image-batch'][i]
        start, end = image_batch.split("--")
        work_times[int(start):int(end) + 1, 0] = metadata['Work-time-in-seconds-1'][i]
        work_times[int(start):int(end) + 1, 1] = metadata['Work-time-in-seconds-2'][i]
        work_times[int(start):int(end) + 1, 2] = metadata['Work-time-in-seconds-3'][i]
    work_time = work_times[:, 0]
    # scipy.stats.pearsonr(clean_labels.numpy(), z[:, 0].numpy())
    # annotated_labels = torch.zeros(len(x), n_labels)
    # annotated_labels[range(len(x)), noised_labels.long()] += 1
    # annotated_labels[range(len(x)), noised_labels2.long()] += 1
    # annotated_labels[range(len(x)), noised_labels3.long()] += 1
    z = torch.cat([n_unique_labels.unsqueeze(-1), work_time.unsqueeze(-1)], dim=-1)
    # z = n_unique_labels.unsqueeze(-1)
    d: torch.Tensor = clean_labels != noised_labels
    full_y = clean_labels.clone()
    y = noised_labels.clone()
    data_masker = DummyDataCorruptionMasker()
    idx = np.random.permutation(len(x))
    return x[idx], full_y[idx], y[idx], z[idx], data_masker, d[idx]


"""
m = NetworkClassifier("e", "", z.shape[-1], 2,
                                             hidden_dims=[32, 64, 64, 32], dropout=0.1, lr=1e-4, wd=0.,
                                             device='cpu', figures_dir='tmp', seed=3,
                                             batch_norm=False)
idx = np.random.permutation(len(x))

train_idx = idx[:250000]
val_idx = idx[250000:30000]
deleted = torch.zeros(len(x))
m.fit(z[train_idx], d[train_idx], deleted[train_idx], z[val_idx], d[val_idx], deleted[val_idx], batch_size=128, epochs=100)
d_hat = m.predict(z).squeeze()[:, 1] >= 0.5
print((d_hat == (z[:, 0] >=3).float()).float().mean().item())
print((d_hat == (d).float()).float().mean().item())
"""


def get_classification_dataset(args) -> ClassificationDataset:
    dataset_name: str = args.dataset_name
    if args.data_type == DataType.Real:
        x, full_y, y, z, data_masker, d = get_real_classification_data(args)
    else:
        raise Exception("does not know how to handle with synthetic data for classification")
    if len(d.shape) == 2:
        marginal_mask_probability = d.any(dim=1).float().mean().item()
    elif len(d.shape) == 1:
        marginal_mask_probability = d.float().mean().item()
    else:
        raise Exception(f"don't know how to handle with len(d.shape)={len(d.shape)}")
    if marginal_mask_probability > 0.6 or marginal_mask_probability < 0.05:
        print(f"warning: marginal mask ratio={marginal_mask_probability}")
    dataset = ClassificationDataset(x, y, z, d, data_masker, dataset_name, args.training_ratio, args.validation_ratio,
                                    args.calibration_ratio, args.device,
                                    args.saved_models_path, args.figures_dir, args.seed,
                                    full_y=full_y)
    print(f"data size: {x.shape[0]}, x_dim: {dataset.x_dim}, y_dim: {dataset.y_dim} z_dim: {dataset.z_dim}")
    return dataset
