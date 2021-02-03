from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import progressbar


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # initialize the loss and the optimizer
        self.criterion = None
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward propagation given a sample."""
        return

    def train(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """Train the model and compute metrics at each epoch.

        :param n_epochs: number of epochs.
        :param train_loader: iterable train set.
        :param validation_loader: iterable validation set.
        :return: the history of the training (accuracy and loss).
        """
        # Get labels of the train and validation set
        train_labels = torch.cat([data[1] for data in train_loader])
        val_labels = torch.cat([data[1] for data in validation_loader])

        history = defaultdict(list)
        for epoch in range(n_epochs):
            print(f"epoch: {epoch + 1}/{n_epochs}")
            train_loss, train_accuracy = 0, 0

            for step, data in progressbar(enumerate(train_loader, 0)):
                # Get the inputs and labels
                inputs, labels = data
                # Reset gradients
                self.optimizer.zero_grad()
                # Perform forward propagation
                outputs = self.forward(inputs)
                # Compute loss and perform back propagation
                loss = self.criterion(outputs, labels)
                loss.backward()
                # Perform the weights' optimization
                self.optimizer.step()
                # Store the step loss and accuracy
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_accuracy += (predicted == labels).float().sum()

            # Compute the epoch training loss and accuracy
            train_loss /= step
            train_accuracy /= train_labels.shape[0]
            history["loss"].append(train_loss)
            history["accuracy"].append(train_accuracy)

            # Compute the epoch validation loss and accuracy
            val_outputs = self.predict(validation_loader)
            val_loss = self.criterion(val_outputs, val_labels)
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == val_labels).float().sum()
            val_accuracy /= val_labels.shape[0]
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)

            # Print statistics at the end of each epoch
            print(
                f"loss: {train_loss:.3f}, val_loss: {val_loss:.3f}",
                f"accuracy: {train_accuracy:.3f},",
                f"val_accuracy: {val_accuracy:.3f} \n",
            )

        return history

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """Compute predictions of the model given samples.

        :param data_loader: iterable data set.
        :return: "probabilities" for each class.
        """
        predictions = []
        # Iterate over the data set without updating gradients
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                predictions.append(self.forward(inputs))

        return torch.cat(predictions)


class NaiveConvNet(BaseNet):
    """Small convolutional network with few parameters (fast but false)."""

    def __init__(self):
        super(NaiveConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward propagation given a sample."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
