from collections import defaultdict
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import get_accuracy, progressbar


class Learner:
    def __init__(self, net: nn.Module):
        self.net = net

    def fit(
        self,
        n_epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> pd.DataFrame:
        """Train the model and compute metrics at each epoch.

        :param n_epochs: number of epochs.
        :param train_loader: iterable train set.
        :param validation_loader: iterable validation set.
        :return: history of the training (accuracy and loss).
        """
        history = defaultdict(list)
        for epoch in range(n_epochs):
            print(f"epoch: {epoch + 1}/{n_epochs}")
            # Set the model in training mode
            self.net.train()
            for step, data in progressbar(enumerate(train_loader, 0)):
                # Get the inputs and labels
                inputs, labels = data

                # Reset gradients
                self.net.optimizer.zero_grad()
                # Perform forward propagation
                outputs = self.net.forward(inputs)
                # Compute loss and perform back propagation
                loss = self.net.criterion(outputs, labels)
                loss.backward()
                # Perform the weights' optimization
                self.net.optimizer.step()

            # Update the learning rate
            self.net.lr_scheduler.step()

            # Compute the epoch training loss and accuracy
            train_outputs, train_labels = self.predict(train_loader)
            train_loss = self.net.criterion(train_outputs, train_labels)
            train_accuracy = get_accuracy(train_outputs, train_labels.numpy())
            history["loss"].append(train_loss.item())
            history["accuracy"].append(train_accuracy)

            # Compute the epoch validation loss and accuracy
            val_outputs, val_labels = self.predict(validation_loader)
            val_loss = self.net.criterion(val_outputs, val_labels)
            val_accuracy = get_accuracy(val_outputs, val_labels.numpy())
            history["val_loss"].append(val_loss.item())
            history["val_accuracy"].append(val_accuracy)

            # Print statistics at the end of each epoch
            print(
                f"loss: {train_loss:.3f}, val_loss: {val_loss:.3f}",
                f"accuracy: {train_accuracy:.3f},",
                f"val_accuracy: {val_accuracy:.3f} \n",
            )

        history_df = pd.DataFrame(history)
        history_df.index.name = "epochs"

        return history_df

    def predict(
        self, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute predictions of the model given samples.

        :param data_loader: iterable data set.
        :return: "probabilities" for each class and the corresponding targets.
        """
        # Set the model in training mode
        self.net.eval()
        outputs, targets = [], []
        # Iterate over the data set without updating gradients
        with torch.no_grad():
            for data in data_loader:
                inputs, labels = data
                outputs.append(self.net.forward(inputs))
                targets.append(labels)

        return torch.cat(outputs), torch.cat(targets)
