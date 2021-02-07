import argparse

from data_processing import Container
from learner import Learner
from models import NaiveConvNet, PreActResNet, ResNet18
from utils import get_accuracy, plot_training_curves

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=str,
        choices=["minicifar_scratch", "minicifar_imagenet"],
        help="Dataset to load",
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["naive_convnet", "preact_resnet", "pretrained_resnet18"],
        help="Model to train",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", "-lr", default=1e-3, help="Learning rate value"
    )
    args = parser.parse_args()

    # Load and process datasets
    container = Container(batch_size=args.batch_size)
    dataset_name = args.dataset
    if dataset_name == "minicifar_scratch":
        container.load_scratch_dataset()
    elif dataset_name == "minicifar_imagenet":
        container.load_imagenet_dataset()

    # Initialize and train the model
    model_name = args.model
    if model_name == "naive_convnet":
        net = NaiveConvNet(n_classes=container.n_classes)
    elif model_name == "preact_resnet":
        net = PreActResNet(n_classes=container.n_classes)
    elif model_name == "pretrained_resnet18":
        net = ResNet18(n_classes=container.n_classes)

    learner = Learner(net, learning_rate=args.learning_rate)

    history = learner.fit(
        n_epochs=args.epochs,
        train_loader=container.train_loader,
        val_loader=container.validation_loader,
    )
    # Plot training curves
    plot_training_curves(
        "loss",
        history,
        f"./logs/{learner.model_name}/training_curves/loss.png",
    )
    plot_training_curves(
        "accuracy",
        history,
        f"./logs/{learner.model_name}/training_curves/accuracy.png",
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
