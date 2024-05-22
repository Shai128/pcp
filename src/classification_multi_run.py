import matplotlib

from data_utils.data_type import DataType
from classification_main import parse_args, main

matplotlib.use('Agg')


def multi_run():
    args = parse_args()
    for seed in list(range(10)):
        args.seed = seed
        print(f"seed: {seed}")
        main(args)


# def run_all():
#     args = parse_args()
#     for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#         for dataset_name in ['bio', 'house']:
#             args.data_type = DataType.Real
#             args.dataset_name = dataset_name
#             args.seed = seed
#             main(args)

        # args.data_type = DataType.Synthetic
        # args.dataset_name = 'partially_linear_syn'
        # args.seed = seed
        # print(f"starting seed: {seed} data: {args.dataset_name}")
        # main(args)
        # print(f"finished seed: {seed} data: {args.dataset_name}")


if __name__ == '__main__':
    multi_run()
    # run_all()
