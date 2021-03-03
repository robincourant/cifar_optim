import argparse

from quantization.quantizer import BinaryQuantizer, HalfQuantizer
from src.container import Container
from src.learner import Learner
from src.models import NaiveConvNet, PreActResNet, ResNet18
from src.utils import get_accuracy, plot_training_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["cifar_scratch", "cifar_imagenet"],
        help="Dataset to load",
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["naive_convnet", "preact_resnet", "pretrained_resnet18"],
        help="Model to train",
    )
    parser.add_argument(
        "--rootdir",
        "-d",
        type=str,
        default=".",
        help="Path to storing directory",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--reduction-rate", "-r", type=int, default=1, help="Reduction rate"
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="Learning rate value",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=1e-3,
        help="Weight decay value",
    )
    parser.add_argument(
        "--momentum", "-m", type=float, default=1e-3, help="Momentum value"
    )
    parser.add_argument(
        "--quantizer",
        "-q",
        type=str,
        default=None,
        choices=["half", "binary"],
        help="Quantizer to use",
    )

    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--logs", "-l", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_false")
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1],
        help="Degree of verbose to use",
    )
    args = parser.parse_args()

    # Load and process datasets
    container = Container(
        rootdir=args.rootdir,
        batch_size=args.batch_size,
        reduction_rate=args.reduction_rate,
    )
    dataset_name = args.dataset
    if dataset_name == "cifar_scratch":
        container.load_scratch_dataset()
    elif dataset_name == "cifar_imagenet":
        container.load_imagenet_dataset()
    else:
        raise ValueError(f"Dataset: {dataset_name} is not known")

    # Initialize and train the model
    model_name = args.model
    if model_name == "naive_convnet":
        net = NaiveConvNet(n_classes=container.n_classes)
    elif model_name == "preact_resnet":
        net = PreActResNet(n_classes=container.n_classes)
    elif model_name == "pretrained_resnet18":
        net = ResNet18(n_classes=container.n_classes)
    else:
        raise ValueError(f"Model: {model_name} is not known")

    quantizer_name = args.quantizer
    if quantizer_name == "half":
        net = HalfQuantizer(net)
    if quantizer_name == "binary":
        net = BinaryQuantizer(net)

    net_params = {
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
    }
    learner = Learner(container, net, args.save, args.logs, **net_params)

    assert not (
        (learner.device == "cuda:0") and (quantizer_name != "half")
    ), "Half quantizer cannot be used without CUDA"

    print("\n")
    learner.get_model_summary()
    print("\n")

    # Fit the model
    history = learner.fit(n_epochs=args.epochs)

    # Plot training curves
    log_dir = f"{container.rootdir}/logs/{learner.model_name}"
    plot_training_curves(
        "loss",
        history,
        f"{log_dir}/training_curves/loss.png",
    )
    plot_training_curves(
        "accuracy",
        history,
        f"{log_dir}/training_curves/accuracy.png",
    )

    train_outputs, train_labels, train_loss = learner.evaluate(
        container.train_loader
    )
    train_accuracy = get_accuracy(train_outputs, train_labels)
    print(f"train_accuracy: {train_accuracy:.3f}")

    test_outputs, test_labels, test_loss = learner.evaluate(
        container.test_loader
    )
    test_accuracy = get_accuracy(test_outputs, test_labels)
    print(f"test_accuracy: {test_accuracy:.3f}")
