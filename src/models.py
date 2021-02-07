import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class NaiveConvNet(nn.Module):
    """Small convolutional network with few parameters (fast but false)."""

    def __init__(self, n_classes=4):
        super(NaiveConvNet, self).__init__()
        self.name = "naive_convnet"
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes)

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
        x = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x
        x = self.conv1(x)
        x = self.conv2(F.relu(self.bn2(x)))
        x += shortcut
        return x


class PreActResNet(nn.Module):
    def __init__(
        self, block=PreActBlock, num_blocks=[2, 2, 2, 2], n_classes=4
    ):
        super(PreActResNet, self).__init__()
        self.name = "preact_resnet"
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet18(nn.Module):
    """Pre-trained resnet model on the ImageNet ILSVRC 2012 dataset."""

    def __init__(self, n_classes=4):
        super(ResNet18, self).__init__()
        self.name = "pretrained_resnet18"
        self.n_classes = n_classes

        # Load the pre-trained model
        self.model = torchvision.models.resnet18(pretrained=True)

        # Freeze weights
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize a fully connected layer to add at the end of the model
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.n_classes)

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self.model(x)
        return x
