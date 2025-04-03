import multiprocessing
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from pruning_utils import (
    custom_loss,
    dataset_path,
    device,
    get_all_conv_layers,
    get_regularizer_value,
)

BATCH_SIZE = 0
train_loader = 0
test_loader = 0


def config(BATCH_SIZE, dataset=1, model=0):
    global train_loader, test_loader
    num_workers = multiprocessing.cpu_count()
    if dataset == 1:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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
            dataset_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            dataset_path, train=False, download=True, transform=transform_val
        )

        train_optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4
        )
        optimize_optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=5e-4
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
            dataset_path, train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.MNIST(
            dataset_path, train=False, download=True, transform=transform_val
        )

        train_optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimize_optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_optimizer, optimize_optimizer


def optimize(
    model, weight_list_per_epoch, epochs, num_filter_pairs_to_prune_per_layer, optimizer
):
    global test_loader, train_loader

    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    print("OPTIMISING MODEL")

    regularizer_value = get_regularizer_value(
        model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer
    )

    criterion = custom_loss(lmbda=0.1, regularizer_value=regularizer_value)
    val_criterion = nn.CrossEntropyLoss()

    print(f"INITIAL REGULARIZER VALUE = {regularizer_value}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total_samples = 0

        progress_bar = tqdm(
            train_loader, desc=f"Optimizing {epoch + 1}/{epochs}", leave=True
        )

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)
            progress_bar.set_postfix(
                loss=train_loss / total_samples,
                acc=100.0 * correct / total_samples,
            )

        train_loss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        history["loss"].append(train_loss)
        history["accuracy"].append(accuracy)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += val_criterion(output, target).item()
                correct += (output.argmax(1) == target).sum().item()

        val_loss /= len(test_loader.dataset)
        val_accuracy = 100.0 * correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"val loss:{val_loss}, val acc:{val_accuracy}")

    print("FINAL REGULARIZER VALUE ", regularizer_value)

    return model, history


def train(model, epochs, optimizer=0):
    global test_loader, train_loader

    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    best_model_path = os.path.join(os.getcwd(), "weights", "temp_best_model.pth")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    conv_layer_names = get_all_conv_layers(model)
    named_modules_dict = dict(model.named_modules())
    weight_list_per_epoch = {layer_name: [] for layer_name in conv_layer_names}

    print("TRAINING MODEL")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total_samples = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True
        )

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()
            total_samples += target.size(0)

            progress_bar.set_postfix(
                loss=train_loss / total_samples,
                acc=100.0 * correct / total_samples,
            )

        train_loss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        history["loss"].append(train_loss)
        history["accuracy"].append(accuracy)

    for layer_name in conv_layer_names:
        if layer_name in named_modules_dict:
            layer = named_modules_dict[layer_name]

            if hasattr(layer, "weight") and layer.weight is not None:
                weight_tensor = layer.weight.data.clone().to(device)
                weight_list_per_epoch[layer_name].append(weight_tensor)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                correct += (output.argmax(1) == target).sum().item()

        val_loss /= len(test_loader.dataset)
        val_accuracy = 100.0 * correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"val loss:{val_loss}, val acc:{val_accuracy}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    # Load the best model before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        print(
            f"Best model weights loaded before returning! best val_acc:{best_val_accuracy}"
        )

    return model, history, weight_list_per_epoch


def evaluate(model):
    global test_loader
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    print("EVALUATING PRE-TRAINED MODEL")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    conv_layer_names = get_all_conv_layers(model)
    named_modules_dict = dict(model.named_modules())

    weight_list_per_epoch = {layer_name: [] for layer_name in conv_layer_names}

    for layer_name in conv_layer_names:
        if layer_name in named_modules_dict:
            layer = named_modules_dict[layer_name]

            if hasattr(layer, "weight") and layer.weight is not None:
                weight_tensor = layer.weight.data.clone().to(device)
                weight_list_per_epoch[layer_name].append(weight_tensor)

    return val_accuracy, val_loss, weight_list_per_epoch
