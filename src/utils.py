import os
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


def progressbar(to_progress: Iterable, n_steps=100, length=60):
    """Display a progress bar when iterating `to_progress`."""

    def show(k):
        """Display the k-th state of a progress bar."""
        x = int(length * k / n_steps)
        sys.stdout.write(
            f"[{'=' * x}{'>' * int(x != length)}{'.' * (length - x - 1)}]"
            + f"{k}/{n_steps}\r",
        )
        sys.stdout.flush()

    show(0)
    for k, item in enumerate(to_progress):
        yield item
        show(k + 1)
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


def get_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute accuracy from output probabiliies."""
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == labels).sum() / predictions.size(0)

    return accuracy
