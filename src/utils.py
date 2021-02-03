import sys
from typing import Iterable


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
