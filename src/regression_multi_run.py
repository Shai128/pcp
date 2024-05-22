import argparse

from data_utils.data_type import DataType
from classification_main import parse_args, main


def multi_run():
    args = parse_args()
    for seed in range(0, 30):
        args.seed = seed
        main(args)


if __name__ == '__main__':
    multi_run()
    # run_all()
