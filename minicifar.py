import numpy as np
from torch.utils.data import Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms


# Generating Mini-CIFAR:
# CIFAR10 is sufficiently large so that training a model up to the state of the
# art performance will take approximately 3 hours on the 1060 GPU available on
# your machine.
# As a result, we will create a "MiniCifar" dataset, based on CIFAR10
# with less classes and exemples.


def train_validation_split(train_size, num_train_examples):
    # Obtain training indices that will be used for validation
    indices = list(range(num_train_examples))
    np.random.shuffle(indices)
    idx_split = int(np.floor(train_size * num_train_examples))
    train_index, valid_index = indices[:idx_split], indices[idx_split:]

    # Define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    return train_sampler, valid_sampler


def generate_subset(dataset, n_classes, reducefactor, n_ex_class_init):
    nb_examples_per_class = int(np.floor(n_ex_class_init / reducefactor))
    # Generate the indices. They are the same for each class, could easily be modified to have different ones. But be careful to keep the random seed!

    indices_split = np.random.RandomState(seed=42).choice(
        n_ex_class_init, nb_examples_per_class, replace=False
    )

    all_indices = []
    for curclas in range(n_classes):
        curtargets = np.where(np.array(dataset.targets) == curclas)
        indices_curclas = curtargets[0]
        indices_subset = indices_curclas[indices_split]
        all_indices.append(indices_subset)
    all_indices = np.hstack(all_indices)

    return Subset(dataset, indices=all_indices)


if __name__ == "__main__":
    train_size = 0.8
    # Must be between 2 and 10
    n_classes_minicifar = 4
    # Reduction factor: 10000/R examples per class for train, 1000/R for test
    R = 5

    # Initialize the data normalization for the untrained model
    normalize_scratch = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )
    # Initialize the data normalization for the pre-trained model
    normalize_forimagenet = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Augment and normalize the train and test sets for the untrained model
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_scratch,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize_scratch,
        ]
    )

    # Normalize and resize the train and test sets for the pre-trained model
    transform_train_imagenet = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize_forimagenet,
        ]
    )
    transform_test_imagenet = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize_forimagenet,
        ]
    )
    rootdir = "./data/cifar10"
    # Load dataset for the untrained model
    c10train_scratch = CIFAR10(
        rootdir, train=True, download=True, transform=transform_train
    )
    c10test_scratch = CIFAR10(
        rootdir, train=False, download=True, transform=transform_test
    )

    # Load dataset for the pre-trained model
    c10train_imagenet = CIFAR10(
        rootdir, train=True, download=True, transform=transform_train_imagenet
    )
    c10test_imagenet = CIFAR10(
        rootdir, train=False, download=True, transform=transform_test_imagenet
    )

    # These dataloader are ready to be used to train for scratch
    minicifar_train_scratch = generate_subset(
        dataset=c10train_scratch,
        n_classes=n_classes_minicifar,
        reducefactor=R,
        n_ex_class_init=5000,
    )
    # Split the train set into a validation and test set for scratch
    num_train_examples_scratch = len(minicifar_train_scratch)
    train_sampler_scratch, valid_sampler_scratch = train_validation_split(
        train_size, num_train_examples_scratch
    )
    minicifar_test_scratch = generate_subset(
        dataset=c10test_scratch,
        n_classes=n_classes_minicifar,
        reducefactor=1,
        n_ex_class_init=1000,
    )

    # These dataloader are ready to be used to train using Transfer Learning
    # from a backbone pretrained on ImageNet
    minicifar_train_imagenet = generate_subset(
        dataset=c10train_imagenet,
        n_classes=n_classes_minicifar,
        reducefactor=R,
        n_ex_class_init=5000,
    )
    # Split the train set into a validation and test set for pre-trained
    num_train_examples_imagenet = len(minicifar_train_imagenet)
    train_sampler_imagenet, valid_sampler_imagenet = train_validation_split(
        train_size, num_train_examples_imagenet
    )
    minicifar_test_imagenet = generate_subset(
        dataset=c10test_imagenet,
        n_classes=n_classes_minicifar,
        reducefactor=1,
        n_ex_class_init=1000,
    )
