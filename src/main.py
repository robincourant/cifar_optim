from data_processing import Container
from learner import Learner
from models import NaiveConvNet, PreActResNet, ResNet18
from utils import get_accuracy, plot_training_curves

if __name__ == "__main__":
    # Load and process datasets
    container = Container()
    container.load_scratch_dataset()
    # container.load_imagenet_dataset()

    # Initialize and train the model
    naive_net = NaiveConvNet(n_classes=container.n_classes)
    preact_net = PreActResNet(n_classes=container.n_classes)
    res_net = ResNet18(n_classes=container.n_classes)

    learner = Learner(preact_net)

    history = learner.fit(
        n_epochs=5,
        train_loader=container.train_loader,
        validation_loader=container.validation_loader,
    )
    # Plot training curves
    plot_training_curves("loss", history)
    plot_training_curves("accuracy", history)

    test_outputs, test_labels = learner.predict(container.test_loader)
    test_accuracy = get_accuracy(test_outputs, test_labels)
    print(test_accuracy)
