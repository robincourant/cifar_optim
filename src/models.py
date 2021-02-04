from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from utils import get_accuracy, progressbar


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        # Initialize the loss and the optimizer
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward propagation given a sample."""
        return

    def train(
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

            # Update the learning rate
            self.lr_scheduler.step()

            # Compute the epoch training loss and accuracy
            train_outputs, train_labels = self.predict(train_loader)
            train_loss = self.criterion(train_outputs, train_labels)
            train_preds = self._get_predictions(train_outputs)
            train_accuracy = get_accuracy(train_preds, train_labels.numpy())
            history["loss"].append(train_loss.item())
            history["accuracy"].append(train_accuracy.item())

            # Compute the epoch validation loss and accuracy
            val_outputs, val_labels = self.predict(validation_loader)
            val_loss = self.criterion(val_outputs, val_labels)
            val_preds = self._get_predictions(val_outputs)
            val_accuracy = get_accuracy(val_preds, val_labels.numpy())
            history["val_loss"].append(val_loss.item())
            history["val_accuracy"].append(val_accuracy.item())

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
        outputs, targets = [], []
        # Iterate over the data set without updating gradients
        with torch.no_grad():
            # loss, accuracy = 0, 0
            for data in data_loader:
                inputs, labels = data
                outputs.append(self.forward(inputs))
                targets.append(labels)

        return torch.cat(outputs), torch.cat(targets)

    def _get_predictions(self, outputs: torch.Tensor) -> List[int]:
        """"""
        _, predictions = torch.max(outputs, 1)
        return predictions.numpy()


class NaiveConvNet(BaseNet):
    """Small convolutional network with few parameters (fast but false)."""

    def __init__(self):
        super(NaiveConvNet, self).__init__()
        self.n_classes = 4

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward propagation given a sample."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(BaseNet):
    def __init__(
        self, block=PreActBlock, num_blocks=[2, 2, 2, 2], num_classes=10
    ):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.6)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet18(BaseNet):
    """Pre-trained resnet model on the ImageNet ILSVRC 2012 dataset."""

    def __init__(self):
        super(ResNet18, self).__init__()
        self.n_classes = 4

        # Load the pre-trained model
        self.model = torchvision.models.resnet18(pretrained=True)

        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize a fully connected layer to add at the end of the model
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.fc.parameters(),
            lr=0.001,
            momentum=0.9,
        )
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self.model(x)
        return x
