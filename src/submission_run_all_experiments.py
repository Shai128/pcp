import argparse
import time
from itertools import product

from run_experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=-1, help='')
args = parser.parse_args()


def cartesian_product(inp):
    if len(inp) == 0:
        return []
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


processes_to_run_in_parallel = 1
max_seed = 20
count = 4
seeds = list(range(0, max_seed))
if args.seed != -1:
    seeds = list(range(args.seed * (max_seed // count), (args.seed + 1) * (max_seed // count)))

real_datasets = ['facebook_1', 'bio', 'house', 'facebook_2', 'meps_19', 'blog']
corruption_types = ['noised_x', 'noised_y', 'missing_x', 'missing_y', 'dispersive_noised_y']
# corruption_types = ['noised_x']
dataset_names = [f"{c}_{d}" for c in corruption_types for d in real_datasets] + ['missing_y_nslm']
real_params = {
    'main_program_name': ['regression_main'],
    'seed': seeds,
    'dataset_name': dataset_names,
    'data_type': ['real'],
    'epochs': [1000],
}

classification_params = {
    'main_program_name': ['classification_main'],
    'seed': seeds,
    'dataset_name': ['missing_y_twins'],
    'data_type': ['real'],
    'epochs': [1000],
}

cifar10_params = {
    'main_program_name': ['classification_main'],
    'seed': seeds,
    'dataset_name': ['noised_y_cifar10'],
    'data_type': ['real'],
    'epochs': [50],
    'bs': [32],
    'training_ratio': [0.5],
    'validation_ratio': [0.1],
    'calibration_ratio': [0.2],
    'alpha': [0.1],
}
cifar10c_params = {
    'main_program_name': ['classification_main'],
    'seed': seeds,
    'dataset_name': ['noised_y_cifar10c_adversarial', 'noised_y_cifar10c'],
    'data_type': ['real'],
    'epochs': [50],
    'bs': [32],
    'training_ratio': [0.5],
    'validation_ratio': [0.1],
    'calibration_ratio': [0.2],
    'alpha': [0.2],
}
ihdp_params = {
    'main_program_name': ['jackknife_experiment'],
    'seed': [0],
    'dataset_name': ['missing_y_ihdp'],
    'data_type': ['real'],
}

params = list(cartesian_product(real_params)) + list(cartesian_product(cifar10_params)) + \
         list(cartesian_product(cifar10c_params)) + list(cartesian_product(classification_params)) + \
         list(cartesian_product(ihdp_params))


processes_to_run_in_parallel = min(processes_to_run_in_parallel, len(params))
run_on_slurm = False
cpus = 2
gpus = 0
if __name__ == '__main__':

    print("jobs to do: ", len(params))
    # initializing processes_to_run_in_parallel workers
    workers = []
    jobs_finished_so_far = 0
    assert len(params) >= processes_to_run_in_parallel
    for _ in range(processes_to_run_in_parallel):
        curr_params = params.pop(0)
        main_program_name = curr_params['main_program_name']
        curr_params.pop('main_program_name')
        p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                           cpus=cpus, gpus=gpus)
        workers.append(p)

    # creating a new process when an old one dies
    while len(params) > 0:
        dead_workers_indexes = [i for i in range(len(workers)) if (workers[i].poll() is not None)]
        for i in dead_workers_indexes:
            worker = workers[i]
            worker.communicate()
            jobs_finished_so_far += 1
            if len(params) > 0:
                curr_params = params.pop(0)
                main_program_name = curr_params['main_program_name']
                curr_params.pop('main_program_name')
                p = run_experiment(curr_params, main_program_name, run_on_slurm=run_on_slurm,
                                   cpus=cpus, gpus=gpus)
                workers[i] = p
                if jobs_finished_so_far % processes_to_run_in_parallel == 0:
                    print(f"finished so far: {jobs_finished_so_far}, {len(params)} jobs left")
            time.sleep(10)

    # joining all last proccesses
    for worker in workers:
        worker.communicate()
        jobs_finished_so_far += 1

    print("finished all")
