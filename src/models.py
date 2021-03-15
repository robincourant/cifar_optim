import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class NaiveConvNet(nn.Module):
    """Small convolutional network with few parameters (fast but false)."""

    def __init__(self, n_classes=10):
        super(NaiveConvNet, self).__init__()
        self.quantizer_name = None
        self.name = "naive_convnet"
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.adaptpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward propagation given a sample."""
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.adaptpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
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
        x = F.relu(self.bn2(x))
        x = self.conv2(x)
        x += shortcut
        return x


class PreActResNet(nn.Module):
    def __init__(
        self, block=PreActBlock, num_blocks=[2, 2, 2, 2], n_classes=4, r=1
    ):
        super(PreActResNet, self).__init__()
        self.quantizer_name = None
        self.name = "preact_resnet"
        self.in_planes = int(64 / r)

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(
            block, self.in_planes, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, 2 * self.in_planes, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 2 * self.in_planes, num_blocks[2], stride=2
        )
        self.layer4 = self._make_layer(
            block, 2 * self.in_planes, num_blocks[3], stride=2
        )
        self.linear = nn.Linear(self.in_planes * block.expansion, n_classes)

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


class SmallPreActResNet(nn.Module):
    """
    Benchmark model for pruning evaluation: enable to train a model from
    scratch with the same numbre of parameters after structured pruning
    (currently remove 40% of layer 4 parameters).
    """

    def __init__(
        self, block=PreActBlock, num_blocks=[2, 2, 2, 2], n_classes=10, r=1
    ):
        super(SmallPreActResNet, self).__init__()
        self.quantizer_name = None
        self.name = "small_preact_resnet"
        self.in_planes = int(64 / r)

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(
            block, self.in_planes, num_blocks[0], stride=1
        )
        self.layer2 = self._make_layer(
            block, 2 * self.in_planes, num_blocks[1], stride=2
        )
        self.layer3 = self._make_layer(
            block, 2 * self.in_planes, num_blocks[2], stride=2
        )
        out_planes = int(0.6 * 2 * self.in_planes)
        self.layer4 = self._make_layer(
            block, out_planes, num_blocks[3], stride=2
        )
        self.linear = nn.Linear(out_planes * block.expansion, n_classes)

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

    def __init__(self, n_classes=10):
        super(ResNet18, self).__init__()
        self.quantizer = None
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


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(),
            nn.Conv2d(
                4 * growth_rate,
                growth_rate,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x0):
        x = self.bottleneck(x0)
        x = torch.cat([x, x0], 1)
        return x


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.AvgPool2d(2),
        )

    def forward(self, x):
        x = self.transition(x)
        return x


class DenseNet(nn.Module):
    def __init__(
        self,
        block=Bottleneck,
        nblocks=[6, 12, 24, 16],
        growth_rate=12,
        reduction=0.5,
        n_classes=10,
    ):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.quantizer_name = None
        self.name = "densenet"

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            3, num_planes, kernel_size=3, padding=1, bias=False
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, n_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
