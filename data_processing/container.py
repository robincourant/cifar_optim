from typing import Any, Tuple

import numpy as np
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from data_processing.autoaugment import CIFAR10Policy
from data_processing.cutout import Cutout


class Container:
    def __init__(
        self,
        rootdir: str = ".",
        train_size: float = 0.8,
        n_classes: int = 10,
        batch_size: int = 32,
        reduction_rate: int = 1,
        augmentation: bool = True,
    ):
        self.rootdir = rootdir
        self.train_size = train_size
        self.n_classes = n_classes  # Number of classes to keep
        self.reduction_rate = (
            reduction_rate  # Keep 1000/R training samples and 100/R test
        )
        self.batch_size = batch_size
        self.augmention = augmentation

    def train_validation_split(
        self, n_train_samples: int
    ) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """Generate train and validation samplers."""
        # Obtain training indices that will be used for validation
        indices = list(range(n_train_samples))
        np.random.shuffle(indices)
        idx_split = int(np.floor(self.train_size * n_train_samples))
        train_index, valid_index = indices[:idx_split], indices[idx_split:]

        # Define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_index)
        valid_sampler = SubsetRandomSampler(valid_index)

        return train_sampler, valid_sampler

    def generate_subset(self, dataset: Any, init_class_samples: int) -> Subset:
        """Subsample a dataset."""
        # Compute the number of samples of each class in the subset
        n_class_samples = int(
            np.floor(init_class_samples / self.reduction_rate)
        )

        indices_split = np.random.RandomState(seed=42).choice(
            init_class_samples, n_class_samples, replace=False
        )

        all_indices = []
        for curclas in range(self.n_classes):
            curtargets = np.where(np.array(dataset.targets) == curclas)
            indices_curclas = curtargets[0]
            indices_subset = indices_curclas[indices_split]
            all_indices.append(indices_subset)
        all_indices = np.hstack(all_indices)

        return Subset(dataset, indices=all_indices)

    def load_dataset(
        self,
        train_data_transformer: transforms.Compose,
        test_data_transformer: transforms.Compose,
    ):
        """Load a dataset given a train and test data transformers."""
        # Load and transform the train and test sets
        train_set = CIFAR10(
            f"{self.rootdir}/data",
            train=True,
            download=True,
            transform=train_data_transformer,
        )
        test_set = CIFAR10(
            f"{self.rootdir}/data",
            train=False,
            download=True,
            transform=test_data_transformer,
        )

        # Subsample the train and test sets
        train_subset = self.generate_subset(
            dataset=train_set,
            init_class_samples=5000,
        )
        test_subset = self.generate_subset(
            dataset=test_set,
            init_class_samples=1000,
        )

        # Initialize the train-validation splitter
        n_samples = len(train_subset)
        train_sampler, val_sampler = self.train_validation_split(n_samples)

        # Generate the train, validation and test loaders
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, sampler=train_sampler
        )
        val_loader = DataLoader(
            train_subset, batch_size=self.batch_size, sampler=val_sampler
        )
        test_loader = DataLoader(test_subset, batch_size=self.batch_size)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        print("\nDatasets summary:")
        self.train_set_size = len(train_loader.dataset)
        print(f"train_set_size: {self.train_set_size:,}")
        self.test_set_size = len(test_loader.dataset)
        print(f"train_set_size: {self.test_set_size:,}")
        self.input_shape = val_loader.dataset[0][0].numpy().shape
        print(f"input_shape: {self.input_shape}")

    def load_scratch_dataset(self):
        """Load and process a dataset for untrained models (from scratch)."""
        # Initialize the data normalizer
        normalizer = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )

        # Initialize the data transformers for train and test sets
        if self.augmention:
            train_data_transformer = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),  # Augmentation
                    transforms.RandomHorizontalFlip(),  # Augmentation
                    CIFAR10Policy(),  # Augmentation
                    transforms.ToTensor(),  # Casting
                    Cutout(n_holes=1, length=16),  # Augmentation
                    normalizer,  # Normalization
                ]
            )
        else:
            train_data_transformer = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),  # Augmentation
                    transforms.RandomHorizontalFlip(),  # Augmentation
                    transforms.ToTensor(),
                    normalizer,
                ]
            )

        test_data_transformer = transforms.Compose(
            [
                transforms.ToTensor(),  # Casting
                normalizer,  # Normalization
            ]
        )

        self.load_dataset(train_data_transformer, test_data_transformer)

    def load_imagenet_dataset(self):
        """Load and process a dataset for the imagenet pre-trained models."""
        # Initialize the data normalizer
        normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        # Initialize the data transformers for train and test sets
        data_transformer = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                normalizer,
            ]
        )

        self.load_dataset(data_transformer, data_transformer)
