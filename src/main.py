from data_processing import Container
from models import NaiveConvNet, PreActResNet, ResNet18
from utils import plot_training_curves

if __name__ == "__main__":
    # Load and process datasets
    container = Container()
    # container.load_scratch_dataset()
    container.load_imagenet_dataset()

    # Initialize and train the model
    # model = NaiveConvNet()
    model = ResNet18()

    history = model.train(
        n_epochs=20,
        train_loader=container.train_loader,
        validation_loader=container.validation_loader,
    )
    # Plot training curves
    plot_training_curves("loss", history)
    plot_training_curves("accuracy", history)

    test_prediction = model.predict(container.test_loader)
