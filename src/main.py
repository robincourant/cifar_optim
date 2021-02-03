from data_processing import Container
from models import NaiveConvNet

if __name__ == "__main__":
    container = Container()
    container.load_scratch_dataset()

    model = NaiveConvNet()
    history = model.train(
        n_epochs=10,
        train_loader=container.train_loader,
        validation_loader=container.validation_loader,
    )
