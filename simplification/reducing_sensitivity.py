"""
(DenseNet architectures)
This script computes the sensitivity curves with regard to the variations of
growth-rate, reduction-rate and depth.
"""
import argparse
from collections import defaultdict
import pickle

import torch

from data_processing.container import Container
from src.learner import Learner
from src.models import DenseNet
from src.micronet_score import profile
from src.utils import get_accuracy


def parse_arguments() -> argparse.ArgumentParser:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootdir",
        type=str,
        help="Path to storing directory",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=2, help="Number of epochs"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save figure or not"
    )

    args = parser.parse_args()

    return args


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    container = Container(
        rootdir=args.rootdir,
        batch_size=32,
        reduction_rate=1,
    )
    container.load_scratch_dataset()

    parameters = {
        "nblocks": [
            [4, 10, 22, 14],
            [2, 8, 16, 12],
            [2, 6, 12, 8],
        ],
        "growth_rate": [11, 9, 6],
        "reduction": [0.3, 0.02],
    }
    log_file = open(args.rootdir + "/logs/simplification_logs.txt", "w+")
    simplification_scores = defaultdict(lambda: defaultdict(list))
    for parameter_name, values in parameters.items():
        for current_value in values:
            net = DenseNet(
                n_classes=container.n_classes,
                **{parameter_name: current_value},
            )
            if parameter_name == "nblocks":
                mode_name = f"simple_{parameter_name}_{''.join(str(value) for value in current_value)}"
            else:
                mode_name = f"simple_{parameter_name}_{current_value}"
            # _, n_params, _ = profile(net, (1, 3, 32, 32), 1)
            learner = Learner(
                container,
                net,
                early_stopping=10,
                model_name=mode_name,
                logs=True,
            )

            # Fit and evaluate the simpler model
            _ = learner.fit(n_epochs=args.epochs)
            outputs, labels, loss = learner.evaluate(container.test_loader)
            accuracy = get_accuracy(outputs, labels)

            print(
                f"[{parameter_name}] {current_value} - accuracy: {accuracy}",
            )

            log_file.write(
                f"[{parameter_name}] {current_value} - accuracy: {accuracy} \n"
            )
            simplification_scores[parameter_name]["value"].append(
                current_value
            )
            # simplification_scores[parameter_name]["n_params"].append(n_params)
            simplification_scores[parameter_name]["accuracy"].append(accuracy)

    simplification_scores = default_to_regular(simplification_scores)
    with open(args.rootdir + "/logs/simplification_scores.pickle", "wb") as f:
        pickle.dump(simplification_scores, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(simplification_scores)
    log_file.close()
