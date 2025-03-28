import multiprocessing
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_pruning as tp
from torchvision import datasets, transforms
from tqdm import tqdm

from models.model_resnet50 import resnet50
from utils_name import *

BATCH_SIZE = 128
INPUT_SHAPE = (BATCH_SIZE, 3, 32, 32)
NO_PRUNING_LIMIT = 8
PRUNE_PER_LAYER = [2] * 49


num_workers = multiprocessing.cpu_count()
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(
    dataset_path, train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    dataset_path, train=False, download=True, transform=transform
)

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


def optimize(model, weight_list_per_epoch, epochs, num_filter_pairs_to_prune_per_layer):
    global test_loader, train_loader

    regularizer_value = get_regularizer_value(
        model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer
    )
    print("INITIAL REGULARIZER VALUE ", regularizer_value)

    criterion = custom_loss(lmbda=0.1, regularizer_value=regularizer_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

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

    print("Training model")
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


def logging(model, history=None, log_dict=None):
    global INPUT_SHAPE
    if log_dict is None:
        log_dict = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "total_params": [],
            "total_flops": [],
        }

    conv_layers = get_all_conv_layers(model)
    for layer_name in conv_layers:
        log_dict[f"filters_in_{layer_name}"] = []

    best_acc_index = history["val_accuracy"].index(max(history["val_accuracy"]))
    log_dict["train_loss"].append(history["loss"][best_acc_index])
    log_dict["train_acc"].append(history["accuracy"][best_acc_index])
    log_dict["val_loss"].append(history["val_loss"][best_acc_index])
    log_dict["val_acc"].append(history["val_accuracy"][best_acc_index])

    total_params, total_flops = count_model_params_flops(model, INPUT_SHAPE)
    log_dict["total_params"].append(total_params)
    log_dict["total_flops"].append(total_flops)

    if log_dict is not None:
        print(f"Current FLOPS: {total_flops}, Current params : {total_params}")

    # Dynamically log filter counts for all convolutional layers
    for layer_name in log_dict.keys():
        if layer_name.startswith("filters_in_"):
            conv_layer_name = layer_name.replace("filters_in_", "")
            conv_layer = dict(model.named_modules()).get(conv_layer_name)
            if conv_layer:
                log_dict[layer_name].append(conv_layer.out_channels)

    print("Validation accuracy ", max(history["val_accuracy"]))

    return log_dict


model = resnet50().to(device)
DG = tp.DependencyGraph().build_dependency(
    model, example_inputs=torch.randn(INPUT_SHAPE).to(device)
)

print("MODEL INITIALIZED AND WEIGHTS LOADED")
validation_accuracy, validation_loss, weight_list_per_epoch = evaluate(model)

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
history["accuracy"].append(validation_accuracy)
history["loss"].append(validation_loss)
history["val_accuracy"].append(validation_accuracy)
history["val_loss"].append(validation_loss)
log_dict = logging(model, history)

max_val_acc = validation_accuracy
count = 0

print("STARTED PRUNING PROCESS")

iterations_without_pruning = 0
initial_parameters = sum(p.numel() for p in model.parameters())

while validation_accuracy - max_val_acc >= -1:
    current_parameters = sum(p.numel() for p in model.parameters())
    print("ITERATION {} ".format(count + 1))
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy

    print(f"MAX VALIDATION ACCURACY = {max_val_acc}")

    if count < 1:
        optimize(model, weight_list_per_epoch, 0, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 2:
        optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 3:
        optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 4:
        optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 5:
        optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 10:
        optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    else:
        optimize(model, weight_list_per_epoch, 10, PRUNE_PER_LAYER)
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 10)

    a, b = count_model_params_flops(model, INPUT_SHAPE)
    print(a, b)

    if current_parameters < initial_parameters:
        initial_parameters = current_parameters
        iterations_without_pruning = 0
    else:
        iterations_without_pruning += 1

    if iterations_without_pruning >= NO_PRUNING_LIMIT:
        print(f"STOPPING EARLY DUE TO NO PRUNING AFTER {NO_PRUNING_LIMIT} ITERATIONS")
        break

    validation_accuracy = max(history["val_accuracy"])
    log_dict = logging(model, history, log_dict)
    print(
        "VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(
            count + 1, validation_accuracy
        )
    )
    count += 1

print(model)

model, history, weight_list_per_epoch = train(model, 30, learning_rate=0.001)
log_dict = logging(model, history, log_dict)

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join(".", "results", "vgg16_cifar10.csv"))
