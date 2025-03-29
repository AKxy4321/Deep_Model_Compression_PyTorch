import multiprocessing
import os
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # Add parent directory to sys.path

from pruning_utils import dataset_path

num_workers = multiprocessing.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_config(batch_size=None, dataset=1):
    global train_loader, test_loader
    if dataset == 1:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615)),
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=dataset_path, train=True, download=True, transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=dataset_path, train=False, download=True, transform=transform_val
        )

    elif dataset == 0:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3015)),
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3015)),
            ]
        )

        train_dataset = datasets.MNIST(
            root=dataset_path, train=True, download=True, transform=transform_train
        )

        test_dataset = datasets.MNIST(
            root=dataset_path, train=False, download=True, transform=transform_val
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
