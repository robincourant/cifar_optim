import os
import sys
from time import time, strftime, gmtime
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def progressbar(to_progress: Iterable, n_steps=100, length=60, verbose=True):
    """Display a progress bar when iterating `to_progress`."""

    def show(k: int, cumulative_time: int):
        """Display the k-th state of a progress bar."""
        x = int(length * k / n_steps)

        current_step_time = strftime("%H:%M:%S", gmtime(cumulative_time))
        approx_total_time = strftime(
            "%H:%M:%S", gmtime((n_steps * cumulative_time) / (k + 1))
        )
        sys.stdout.write(
            f"[{'=' * x}{'>' * int(x != length)}{'.' * (length - x - 1)}]"
            + f"{k}/{n_steps} ETA: {current_step_time}/{approx_total_time}\r",
        )
        sys.stdout.flush()

    cumulative_time = 0
    if verbose:
        show(0, cumulative_time)

    for k, item in enumerate(to_progress):
        t0 = time()
        yield item
        cumulative_time += time() - t0
        if verbose:
            show(k + 1, cumulative_time)
        else:
            sys.stdout.write("")

    sys.stdout.write("\n")
    sys.stdout.flush()


def plot_training_curves(metric: str, history: pd.DataFrame, path: str):
    """Plot the evolution of a train and validation metric over the epochs.

    :param metric: name of the metric to plot.
    :param history: history of the training (accuracy and loss).
    :param path: if provided save figure at this path.
    """
    sns.lineplot(
        x="epochs",
        y=metric,
        data=history,
        label="train",
        color="#e41a1c",
    )
    sns.lineplot(
        x="epochs",
        y="val_" + metric,
        data=history,
        label="validation",
        color="#377eb8",
    )
    plt.title(f"{metric} over training epochs")

    if path:
        directory_name = os.path.dirname(path)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        plt.savefig(path)

    plt.show()


def plot_sensitivity_curves(sensitivity_data: pd.DataFrame, path: str):
    """Plot the pruning sensitivity for each layer for different pruning rates.

    :param sensitivity_data: pruning sensitivity for different pruning rates.
    :param path: if provided save figure at this path.
    """
    sns.set_context("paper", rc={"lines.linewidth": 1, "lines.markersize": 10})
    sns.catplot(
        x="pruning_rate",
        y="accuracy",
        hue="layers",
        data=sensitivity_data,
        kind="point",
    )
    if path:
        directory_name = os.path.dirname(path)
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        plt.savefig(path)

    plt.show()


def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from output probabiliies."""
    _, predictions = torch.max(outputs, 1)
    accuracy = ((predictions == labels).sum() / predictions.size(0)).item()

    return accuracy
