import sys

import matplotlib
import numpy as np
import pandas as pd
import os
from typing import List
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numbers
import re

matplotlib.rcParams["legend.framealpha"] = 1


# matplotlib.rc('font', **{'size': 20})


def is_number(x):
    return isinstance(x, numbers.Number)


figure_save_dir = 'paper_figures'


def save_figure(save_dir, file_name):
    os.makedirs(os.path.join(figure_save_dir, save_dir), exist_ok=True)
    full_save_path = os.path.join(figure_save_dir, save_dir, file_name)
    plt.savefig(full_save_path, dpi=300, bbox_inches='tight')


results_base_path = '../src/results'
seeds = 100


def read_method_results_aux(folder_path, seeds=20, apply_mean=True, display_errors=False):
    df = pd.DataFrame()

    for seed in range(seeds):
        save_path = f"{folder_path}/seed={seed}.csv"
        try:
            seed_df = pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')

            if 'coverage' in seed_df and abs(
                    seed_df['coverage'].item() - 0) < 0.01:
                # print(f"{folder_path}/seed={seed}.csv has 0 coverage")
                if np.isnan(seed_df['average length']).any():
                    print(
                        f"{folder_path}/seed={seed}.csv has invalid average length. the value is: {seed_df['average length'].item()}")
                    display(seed_df)
                    # print("got here")
                    continue
            if '(miscoverage streak) average length' in df.columns and \
                    np.isnan(seed_df['(miscoverage streak) average length']).any():
                print(
                    f"{folder_path}/seed={seed}.csv has invalid (miscoverage streak) average length")
                print("the value is: ", seed_df['(miscoverage streak) average length'].item())
                display(seed_df)

            df = pd.concat([df, seed_df], axis=0)
        except Exception as e:
            # print("got an exception")
            if display_errors:
                print(e)
    if len(df) == 0:
        # print(f"{folder_path} had 0 an error")
        save_path = f"{folder_path}/seed=0.csv"
        pd.read_csv(save_path).drop(['Unnamed: 0'], axis=1, errors='ignore')  # raises an exception
        raise Exception(f"could not find results in path {folder_path}")

    if apply_mean:
        df = df.apply(np.mean).to_frame().T

    return df


def read_method_results(base_path: str, dataset_name: str, method_name: str, seeds=20, apply_mean=True,
                        display_errors=False):
    full_folder_path = os.path.join(base_path, dataset_name, method_name)
    df = read_method_results_aux(full_folder_path, seeds, apply_mean, display_errors=display_errors)
    return df


# pd.set_option('display.max_columns', None)
imputations = ['linear', 'partially_linear', 'dml', 'full', 'full_with_linear', 'indep_partially_linear']
errors_methods = ['marginal', 'kmeans_clustering', 'linear_clustering', 'nn', 'CVAE']
method_name_to_display_name = {}
for cal in ['cqr', 'hps']:
    for model in ['cnn', 'qr', 'xgb_qr', 'xgb_classifier']:
        for masker in [
            'cnn_use_z=False', 'cnn_use_z=True',
            'network_use_z=False', 'network_use_z=True',
            'xgb_use_z=False', 'xgb_use_z=True',
            'rf_use_z=False', 'rf_use_z=True', 'oracle']:
            method_name_to_display_name = {
                **method_name_to_display_name,
                f'{model}_Dummy': 'Uncalibrated',
                f"{model}_{cal}": f'Naive CP',
                f"{model}_{cal}_ignore_masked": r'Naive CP (only clean)',
                f"{model}_oracle_{cal}": f'clean Y {cal} ',
                f"{model}_weighted_{cal}_{masker}_masker": f'weighted_{masker}_masker',
                f"{model}_two_staged_{cal}_{masker}_masker": f'Two_Staged_{masker}_masker',

                f"{model}_pcp_{cal}_{masker}_masker": f'PCP_{masker}_masker',
            }
for model in ['qr', 'xgb']:
    for impuation in imputations:
        imputation_display = impuation.replace("full_with_linear", "full + linear")
        method_name_to_display_name[f'{model}_{impuation}_imputation_cqr_calibration'] = imputation_display
        for error_method in errors_methods:
            method_name_to_display_name[
                f'{model}_{impuation}_with_{error_method}_error_sampler_imputation_cqr_calibration'] = \
                f"{imputation_display} with {error_method} errors"

for key in method_name_to_display_name.keys():
    method_name_to_display_name[key] = method_name_to_display_name[key].replace("_", " ")

methods = list(method_name_to_display_name.keys())


def read_methods_results(base_path: str, dataset_name: str, method_names: List[str], seeds=20, display_errors=False,
                         apply_mean=True):
    total_df = pd.DataFrame()
    for method_name in method_names:
        try:
            full_folder_path = os.path.join(base_path, dataset_name, method_name)
            df = read_method_results_aux(full_folder_path, seeds, apply_mean=apply_mean, display_errors=display_errors)
            df['Method'] = get_method_display_name(method_name)
            total_df = pd.concat([total_df, df])
        except Exception as e:
            if display_errors:
                print(f"got error while trying to read method {method_name}. error: {e}")
    return total_df


def get_method_display_name(method_name):
    if method_name in method_name_to_display_name:
        return method_name_to_display_name[method_name]
    else:
        return method_name


def method_to_error_type(method):
    if 'marginal' in method or 'with errors' in method:
        return 'marginal'
    elif 'kmeans' in method:
        return 'kmeans clustering'
    elif 'linear clustering' in method:
        return 'linear clustering'
    elif 'cvae' in method:
        return 'cvae'
    elif 'nn' in method:
        return 'qr'
    elif "errors" not in method:
        return 'none'
    else:
        raise Exception(f"don't know how to handle with method: {method}")


def method_to_display_name(method):
    errors_txt = re.search(r' with.*errors', method)
    if errors_txt is None:
        return method.replace("imputation", "")
    return method.replace(errors_txt.group(), "").replace("imputation", "")


def process_methods_df(total_df):
    if len(total_df) == 0:
        raise Exception("no data")
    cols = list(total_df.columns)
    for col in cols:
        if 'coverage' in col:
            total_df[col] *= 100
        if is_number(total_df[col].iloc[0]):
            total_df[col] = np.round(total_df[col], 2)

    return total_df


def process_methods_names(total_df):
    if len(total_df) == 0:
        raise Exception("no data")
    total_df['Error'] = total_df['Method'].apply(method_to_error_type)
    total_df['Method'] = total_df['Method'].apply(method_to_display_name)
    return total_df


masker_name_to_display_name_old = {
    'oracle masker': 'oracle',
    'network use z=False masker': "without z",
    'network use z=True masker': "with z",
    'xgb use z=False masker': "without z",
    'xgb use z=True masker': "with z",
    'cnn use z=False masker': "without z",
    'cnn use z=True masker': "with z",
    'rf use z=False masker': "without z",
    'rf use z=True masker': "with z",

}
masker_name_to_display_name = {
    k: v.replace("without z", "estimated from X").replace("with z", "estimated from Z")
    for k, v in masker_name_to_display_name_old.items()
}


def get_masker_from_method_name_old(method_name):
    for k in masker_name_to_display_name_old:
        if k in method_name:
            return masker_name_to_display_name_old[k]
    return '-'

def get_masker_from_method_name(method_name):
    for k in masker_name_to_display_name:
        if k in method_name:
            return masker_name_to_display_name[k]
    return '-'


def remove_masker_name_from_method_name(method_name):
    for k in masker_name_to_display_name:
        method_name = method_name.replace(' ' + k, "")
    return method_name

def get_display_method_name(method_name):
    masker = get_masker_from_method_name_old(method_name)
    if 'weighted' in method_name:
        if masker == 'with z' or masker == 'oracle':
            return "Infeasible WCP"
        else:
            return 'Naive WCP'
    if 'Two Staged' in method_name:
        return "Two Staged CP"
    if 'PCP' in method_name:
        return 'Privileged CP'
    return method_name


color_palette = {
                "uncalibrated": "hotpink",
                "Uncalibrated": "hotpink",
                'Naive CP': "b",
                 'Naive CP\n(clean + noisy)': "b",
                 'Naive CP\n(only clean)': "m",
                 'Naive JP': "b",
                 'Naive WCP': "r",
                 'Naive JAWS': "r",
                 'Two Staged CP': "y",
                 'Infeasible WCP': "g",
                 'Infeasible JAWS': "g",
                 'Privileged CP': "c",
                 }

methods_order = [
                'uncalibrated',
                'Uncalibrated',
                'Naive CP', 'Naive CP\n(clean + noisy)', 'Naive CP\n(only clean)',
                 'Naive JP',
                 'Naive WCP', 'Naive JAWS',
                 'Two Staged CP', 'Infeasible WCP', 'Infeasible JAWS',
                 'Privileged CP']


def get_display_method_name_for_full_figure(method_name):
    masker = get_masker_from_method_name(method_name)
    if 'uncalbirated' in method_name:
        return 'Uncalbirated'
    if 'Naive CP' in method_name and 'only clean' not in method_name:
        return 'Naive CP\n(noisy+clean)'
    if 'Naive CP' in method_name and 'only clean' in method_name:
        return 'Naive CP\n(only clean)'
    if 'weighted' in method_name:
        return 'Weighted CP'
    if 'Two Staged' in method_name:
        return "Two Staged CP"
    if 'PCP' in method_name:
        return 'Privileged CP'
    return method_name


original_dataset_name_to_display_name = {
    'ihdp': 'IHDP',
    'twins': 'Twins',
    'nslm': 'NSLM',
    'cifar10': 'CIFAR-10N',
    'cifar10c': 'CIFAR-10C',
    'cifar10c_adversarial': 'CIFAR-10C',
}


original_dataset_name_to_display_task = {
    'ihdp': 'Causal inference experiment',
    'twins': 'Causal inference experiment',
    'nslm': 'Causal inference experiment',
    'cifar10': 'Noisy response experiment',
    'cifar10c': 'Noisy response experiment',
    'cifar10c_adversarial': 'Adversarial noisy response experiment',
}


def plot_full_figure(original_dataset_name, dataset, nominal_coverage_level=90,
                     is_classification=False,
                     methods_to_remove=None,
                     **kwargs):
    total_df = read_methods_results(results_base_path, dataset, methods, apply_mean=False, seeds=seeds)
    total_df = process_methods_names(process_methods_df(total_df))
    methods_to_keep = ['Uncalibrated', 'Naive', 'weighted', 'Two Staged', 'PCP']

    methods_to_exclude = []

    def keep_method(method_name):
        return any([a in method_name for a in methods_to_keep]) and not any(
            [a in method_name for a in methods_to_exclude])

    total_df = total_df[total_df['Method'].apply(keep_method)]
    total_df['Masker'] = total_df['Method'].apply(lambda x: get_masker_from_method_name(x))
    total_df['Method'] = total_df['Method'].apply(lambda x: remove_masker_name_from_method_name(x))
    total_df['Method'] = total_df['Method'].apply(get_display_method_name_for_full_figure)
    if methods_to_remove is not None:
        for method in methods_to_remove:
            total_df = total_df[total_df['Method'] != method]
    if len(list(filter(lambda x: 'Naive CP' in x, total_df['Method'].unique()))) == 1:
        total_df['Method'] = total_df['Method'].apply(lambda x: 'Naive CP' if 'Naive CP' in x else x)

    # sns.set(font_scale=2)
    sns.set_theme(context='paper', style={
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.bottom': True,
        'xtick.top': False,
        'ytick.left': True,
        'ytick.right': False,
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.grid': True,
    },
                  font_scale=2.5, rc={'figure.figsize': (15, 3)})

    ax = sns.boxplot(data=total_df, x='Method', y='full y2 coverage',
                     # palette=color_palette, order=curr_methods_order,
                     hue='Masker', **kwargs)
    ax.legend().set_title("Corruption probability")
    plt.ylabel("Coverage rate")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.axhline(y=nominal_coverage_level, color='r', linestyle='--')

    data_display_name = original_dataset_name_to_display_name[original_dataset_name]
    task_display_name = original_dataset_name_to_display_task[original_dataset_name]
    # plt.title(f"{task_display_name} - {data_display_name} dataset")
    save_figure(original_dataset_name, "full_coverage.png")
    plt.show()

    sns.boxplot(data=total_df, x='Method', y='y2 length',
                #             palette=color_palette, order=curr_methods_order,
                hue='Masker', **kwargs)
    plt.legend().remove()
    if is_classification:
        plt.ylabel("Set size")
    else:
        plt.ylabel("Interval length")
    # plt.title(f"{task_display_name} - {data_display_name} dataset")
    save_figure(original_dataset_name, "full_length.png")
    plt.show()
