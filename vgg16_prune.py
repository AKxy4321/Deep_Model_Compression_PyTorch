from torchvision import datasets, transforms
import torch.nn.functional as F
from vgg16_model import vgg16
import torch.optim as optim
import torch_pruning as tp
import multiprocessing
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from utils import *
import torch
import os


BATCH_SIZE = 128
INPUT_SHAPE = (BATCH_SIZE, 3, 32, 32)
NO_PRUNING_LIMIT = 8
PRUNE_PER_LAYER = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


num_workers = multiprocessing.cpu_count()
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.CIFAR10(
    "./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    "./data", train=False, download=True, transform=transform
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
            train_loader, desc=f"Optimizing {epoch+1}/{epochs}", leave=True
        )

        for data, target in progress_bar:
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
    conv_indices = get_all_conv_layers(model)
    weight_list_per_epoch = [[] for _ in conv_indices]

    print("Training model")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for data, target in progress_bar:
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

        for i, layer_idx in enumerate(conv_indices):
            weight_list_per_epoch[i].append(model[layer_idx].weight.data.clone().cpu())

        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
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
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")

    conv_indices = get_all_conv_layers(model)
    weight_list_per_epoch = [[] for _ in conv_indices]
    for i, layer_idx in enumerate(conv_indices):
        weight_list_per_epoch[i].append(model[layer_idx].weight.data.clone().cpu())
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
            "filters_in_conv1": [],
            "filters_in_conv2": [],
        }
        initial_params, initial_flops = count_model_params_flops(model, INPUT_SHAPE)
        print(f"INITIAL FLOPS: {initial_flops}, INITIAL params : {initial_params}")

    best_acc_index = history["val_accuracy"].index(max(history["val_accuracy"]))
    log_dict["train_loss"].append(history["loss"][best_acc_index])
    log_dict["train_acc"].append(history["accuracy"][best_acc_index])
    log_dict["val_loss"].append(history["val_loss"][best_acc_index])
    log_dict["val_acc"].append(history["val_accuracy"][best_acc_index])
    a, b = count_model_params_flops(model, INPUT_SHAPE)
    log_dict["total_params"].append(a)
    log_dict["total_flops"].append(b)
    if log_dict is not None:
        print(f"Current FLOPS: {b}, Current params : {a}")
    log_dict["filters_in_conv1"].append(model[0].out_channels)
    log_dict["filters_in_conv2"].append(model[2].out_channels)

    print("Validation accuracy ", max(history["val_accuracy"]))

    return log_dict


model = vgg16()
# model.load_state_dict(
#     torch.load(os.path.join(os.getcwd(), "models", "lenet_best.pth"), weights_only=True)
# )
DG = tp.DependencyGraph().build_dependency(
    model, example_inputs=torch.randn(INPUT_SHAPE)
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
a, b = count_model_params_flops(model, INPUT_SHAPE)
print(a, b)

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
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)
        print(model)

    elif count < 2:
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 3:
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 4:
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 5:
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 10:
        optimize(
            model, weight_list_per_epoch, 1, PRUNE_PER_LAYER
        )
        model = delete_filters(
            model,
            weight_list_per_epoch,
            PRUNE_PER_LAYER,
            DG=DG,
            input_shape=INPUT_SHAPE,
        )
        model, history, weight_list_per_epoch = train(model, 1)

    else:
        optimize(
            model, weight_list_per_epoch, 10, PRUNE_PER_LAYER
        )
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
