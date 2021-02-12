from collections import defaultdict
import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from utils import get_accuracy, progressbar
from data_processing import Container


class Learner:
    def __init__(
        self,
        container: Container,
        net: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 5e-4,
        momentum: float = 0.9,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.container = container
        self.net = net.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        self.n_early_stopping = 10

        self.model_name = (
            f"{self.net.name}_lr{learning_rate}_wd{weight_decay}_m{momentum}"
        )
        self.writer = SummaryWriter(
            f"{self.container.rootdir}/logs/{self.model_name}"
        )

    def fit(
        self,
        n_epochs: int,
    ) -> pd.DataFrame:
        """Train the model and compute metrics at each epoch.

        :param n_epochs: number of epochs.
        :param train_loader: iterable train set.
        :param val_loader: iterable validation set.
        :return: history of the training (accuracy and loss).
        """
        history = defaultdict(list)
        best_accuracy = -1

        # Cosine learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs
        )

        for epoch in range(n_epochs):
            self.current_epoch = epoch
            print(f"epoch: {epoch + 1}/{n_epochs}")

            # Training step
            _, _, _ = self._train(self.container.train_loader)
            # Update the learning rate
            self.lr_scheduler.step()

            # Evaluation step
            if self.net.quantizer == "binary":
                self.net.binarize_params()
            train_outputs, train_labels, train_loss = self.evaluate(
                self.container.train_loader
            )
            val_outputs, val_labels, val_loss = self.evaluate(
                self.container.val_loader
            )
            # Compute training and validation accuracy
            train_accuracy = get_accuracy(train_outputs, train_labels)
            val_accuracy = get_accuracy(val_outputs, val_labels)

            # Store metrics
            history["loss"].append(train_loss)
            history["accuracy"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_accuracy)
            self._write_summary(
                train_loss, train_accuracy, val_loss, val_accuracy
            )

            # Save the best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self._save()

            if self.net.quantizer == "binary":
                self.net.restore_params()

            # Print statistics at the end of each epoch
            print(
                f"loss: {train_loss:.3f}, val_loss: {val_loss:.3f}",
                f"accuracy: {train_accuracy:.3f},",
                f"val_accuracy: {val_accuracy:.3f} \n",
            )

            # Early stopping
            if (self.current_epoch >= self.n_early_stopping) and (
                history["val_loss"][-1]
                > history["val_loss"][-self.n_early_stopping]
            ):
                print("Early stopping criterion reached")
                break

        history_df = pd.DataFrame(history)
        history_df.index.name = "epochs"
        self.writer.close()
        # Load the best model
        self.net = self.load(
            f"{self.container.rootdir}/models/{self.model_name}.pth"
        )

        return history_df

    def _write_summary(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
    ):
        """Write training and vlaidation loss in tensoboard."""
        self.writer.add_scalar(
            "loss/train_epoch",
            train_loss,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "loss/val_epoch",
            val_loss,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "accuracy/train_epoch",
            train_accuracy,
            self.current_epoch,
        )
        self.writer.add_scalar(
            "accuracy/val_epoch",
            val_accuracy,
            self.current_epoch,
        )

        self.writer.flush()

    def _train(
        self, train_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Train the model over the given samples.

        :param train_loader: iterable data set.
        :return: "probabilities" for each class and targets and loss.
        """
        # Set the model in training mode
        self.net.train()
        train_outputs, train_labels, train_loss = [], [], []
        for step, (inputs, labels) in progressbar(
            enumerate(train_loader), n_steps=len(train_loader)
        ):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # Reset gradients
            self.optimizer.zero_grad()
            if self.net.quantizer == "binary":
                self.net.binarize_params()
            # Perform forward propagation
            outputs = self.net(inputs)
            if self.net.quantizer == "binary":
                self.net.restore_params()
            # Compute loss and perform back propagation
            loss = self.criterion(outputs, labels)
            loss.backward()
            # Perform the weights' optimization
            self.optimizer.step()
            if self.net.quantizer == "binary":
                self.net.clip_params()

            train_outputs.append(outputs)
            train_labels.append(labels)
            train_loss.append(loss)
            self.writer.add_scalar(
                "loss/train_step",
                loss,
                self.current_epoch * len(train_loader) + step,
            )

        train_loss = torch.mean(torch.stack(train_loss)).item()
        self.writer.flush()

        return torch.cat(train_outputs), torch.cat(train_labels), train_loss

    def evaluate(
        self, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute predictions of the model given samples.

        :param data_loader: iterable data set.
        :return: "probabilities" for each class and targets and loss.
        """
        # Set the model in evaluation mode
        self.net.eval()
        eval_outputs, eval_labels, eval_loss = [], [], []
        # Iterate over the data set without updating gradients
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)

                eval_outputs.append(outputs)
                eval_labels.append(labels)
                eval_loss.append(loss)

            eval_loss = torch.mean(torch.stack(eval_loss)).item()

        return torch.cat(eval_outputs), torch.cat(eval_labels), eval_loss

    def _save(self):
        """Save a model."""
        if not os.path.exists(f"{self.container.rootdir}/models"):
            os.makedirs(f"{self.container.rootdir}/models")
        saving_path = f"{self.container.rootdir}/models/{self.model_name}.pth"
        torch.save(self.net, saving_path)
        print("Best model saved")

    def load(self, model_path: str):
        """TODO: Load a saved model."""
        return torch.load(model_path)

    def get_model_summary(self):
        """Print summary of the model."""
        summary(self.net, self.container.input_shape)
