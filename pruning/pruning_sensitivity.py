"""
(Preact-ResNet architectures)
This script computes the sensitivity curves of each layer of the model for
unstructured or structured pruning.
"""

import argparse
from collections import defaultdict

import pandas as pd

from data_processing.container import Container
from simplification.pruner import UnstructuredPruner, StructuredPruner
from src.learner import Learner
from src.models import PreActResNet, SmallPreActResNet
from src.utils import get_accuracy, plot_sensitivity_curves


def parse_arguments() -> argparse.ArgumentParser:
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rootdir",
        type=str,
        help="Path to storing directory",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of the model to load",
    )
    parser.add_argument(
        "pruner",
        type=str,
        choices=["unstructured", "structured"],
        help="Pruner to use",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=2, help="Number of epochs"
    )
    parser.add_argument(
        "--save", "-s", action="store_true", help="Save figure or not"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()

    container = Container(
        rootdir=args.rootdir,
        batch_size=32,
        reduction_rate=1,
    )
    container.load_scratch_dataset()

    model_path = args.rootdir + "/models/" + args.model_name + ".pth"
    net = SmallPreActResNet(n_classes=container.n_classes)
    learner = Learner(container, net)
    learner.get_model_summary()

    if args.pruner == "unstructured":
        Pruner = UnstructuredPruner
    if args.pruner == "structured":
        Pruner = StructuredPruner

    no_retrain_res = defaultdict(list)
    retrain_res = defaultdict(list)

    # Load the model
    learner.load(model_path)
    # Evaluate without retraining
    outputs, labels, loss = learner.evaluate(container.test_loader)
    accuracy = get_accuracy(outputs, labels)
    no_retrain_res[0.0] = [accuracy for _ in range(3)]
    retrain_res[0.0] = [accuracy for _ in range(3)]

    rates = [0.2, 0.4, 0.6, 0.8]
    for current_pruning_rate in rates:
        for k_layer in range(1, 4):
            # Set the pruning rates for each layer
            current_layer = f"dense{k_layer}"
            pruning_rates = {f"dense{k}": 0 for k in range(1, 4)}
            pruning_rates[current_layer] = current_pruning_rate

            # Load the model
            learner.load(model_path)

            # Initialize the pruner and prune the model
            pruner = Pruner(learner)
            pruner.prune_model(pruning_rates)
            if args.pruner == "unstructured":
                # For unstructured pruning look at the sparsity of weights
                pruner.get_sparsity()
            if args.pruner == "structured":
                # For structured pruning look at the number of parameters
                learner.get_model_summary()

            # Evaluate without retraining
            outputs, labels, loss = learner.evaluate(container.test_loader)
            accuracy = get_accuracy(outputs, labels)
            no_retrain_res[current_pruning_rate].append(accuracy)
            # Evaluate after retraining
            _ = learner.fit(n_epochs=args.epochs)
            outputs, labels, loss = learner.evaluate(container.test_loader)
            accuracy = get_accuracy(outputs, labels)
            retrain_res[current_pruning_rate].append(accuracy)
            print("\n")

    # Add the name of corresponding layers
    no_retrain_res["layers"] = ["layer1", "layer2", "layer3"]
    retrain_res["layers"] = ["layer1", "layer2", "layer3"]
    # Add accuracy if the model is fully pruned (random for pruning rate = 1)
    random_accuracy = 1 / container.n_classes
    no_retrain_res[1.0] = [random_accuracy for _ in range(3)]
    retrain_res[1.0] = [random_accuracy for _ in range(3)]

    no_retrain_res_df = pd.DataFrame(no_retrain_res).melt(
        "layers", var_name="pruning_rate", value_name="accuracy"
    )
    retrain_res_df = pd.DataFrame(retrain_res).melt(
        "layers", var_name="pruning_rate", value_name="accuracy"
    )
    # Plot curves
    no_retrain_plot_path = None
    retrain_plot_path = None
    if args.save:
        log_dir = f"{container.rootdir}/logs/{args.model_name}"
        no_retrain_plot_path = f"{log_dir}/sensitivity_curves/no_retrain.png"
        retrain_plot_path = f"{log_dir}/sensitivity_curves/retrain.png"
    print(no_retrain_plot_path)
    plot_sensitivity_curves(no_retrain_res_df, no_retrain_plot_path)
    plot_sensitivity_curves(retrain_res_df, retrain_plot_path)
