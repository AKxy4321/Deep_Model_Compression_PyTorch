import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from pruning_utils import (
    custom_loss,
    device,
    get_all_conv_layers,
    get_regularizer_value,
)


def optimize(model, weight_list_per_epoch, epochs, num_filter_pairs_to_prune_per_layer):
    global test_loader, train_loader

    regularizer_value = get_regularizer_value(
        model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer
    )
    print("INITIAL REGULARIZER VALUE ", regularizer_value)

    criterion = custom_loss(lmbda=0.1, regularizer_value=regularizer_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    print("OPTIMISING MODEL")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        progress_bar = tqdm(
            train_loader, desc=f"Optimizing {epoch + 1}/{epochs}", leave=True
        )

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            progress_bar.set_postfix(loss=train_loss / len(train_loader.dataset))

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
                val_loss += criterion(target, output).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(test_loader.dataset)
        val_accuracy = 100.0 * correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        progress_bar.set_postfix(val_loss=val_loss, val_acc=val_accuracy)

    print("FINAL REGULARIZER VALUE ", regularizer_value)

    return model, history


def train(model, epochs, learning_rate=0.001):
    global test_loader, train_loader

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    conv_layer_names = get_all_conv_layers(model)
    named_modules_dict = dict(model.named_modules())
    weight_list_per_epoch = {layer_name: [] for layer_name in conv_layer_names}

    print("TRAINING MODEL")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
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
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            progress_bar.set_postfix(loss=train_loss / len(train_loader.dataset))

        train_loss /= len(train_loader.dataset)
        accuracy = 100.0 * correct / len(train_loader.dataset)
        history["loss"].append(train_loss)
        history["accuracy"].append(accuracy)

    for layer_name in conv_layer_names:
        if layer_name in named_modules_dict:
            layer = named_modules_dict[layer_name]

            if hasattr(layer, "weight") and layer.weight is not None:
                weight_tensor = layer.weight.data.clone().cpu()
                weight_list_per_epoch[layer_name].append(weight_tensor)

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(test_loader.dataset)
        val_accuracy = 100.0 * correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        progress_bar.set_postfix(val_loss=val_loss, val_acc=val_accuracy)

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
                weight_tensor = layer.weight.data.clone().cpu()
                weight_list_per_epoch[layer_name].append(weight_tensor)

    return val_accuracy, val_loss, weight_list_per_epoch
