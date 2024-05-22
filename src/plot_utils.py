import os
from typing import List

from matplotlib import pyplot as plt

from utils import create_folder_if_it_doesnt_exist


def display_plot(ys: List, xs: List = None, title=None, labels=None, x_label: str = None, y_label: str = None,
                 save_dir: str = None):
    fig, ax = plt.subplots()
    assert len(ys) > 0
    if xs is None:
        xs = [None] * len(ys)
    assert len(xs) == len(ys)
    if labels is None:
        labels = [None] * len(xs)

    for x, y, label in zip(xs, ys, labels):
        args = {}
        if label is not None:
            args['label'] = label
        if x is None:
            ax.plot(y, **args)
        else:
            ax.plot(x, y, **args)

    ax.set_title(title)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if labels[0] is not None:
        ax.legend()
    if save_dir is not None:
        create_folder_if_it_doesnt_exist(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
