import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pandas as pd
import os
from utils import *
from lenet import LeNet

def optimize(model, weight_list_per_epoch, epochs, percentage, first_time):
    """
    Arguments:
        model: initial model
        weight_list_per_epoch: weight tensors at every epoch
        epochs: number of epochs to be trained on custom regularizer
        percentage: percentage of filters to be pruned
        first_time: type bool
    Return:
        model: optimized model
        history: accuracies and losses over the process
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(".", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    regularizer_value = my_get_regularizer_value(
        model, weight_list_per_epoch, percentage, first_time
    )
    print("INITIAL REGULARIZER VALUE ", regularizer_value)
    criterion = custom_loss(lmbda=0.1, regularizer_value=regularizer_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
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

    print("FINAL REGULARIZER VALUE ", regularizer_value)
    return model, history


def my_get_regularizer_value(model, weight_list_per_epoch, percentage, first_time):
    """
    Arguments:
        model: initial model
        weight_list_per_epoch: weight tensors at every epoch
        percentage: percentage of filter to be pruned
        first_time: type bool
    Return:
        regularizer_value
    """
    _, filter_pairs = find_pruning_indices(
        model, weight_list_per_epoch, first_time, percentage
    )
    l1_norms = my_get_l1_norms_filters(model, first_time)
    regularizer_value = 0
    for layer_index, layer in enumerate(filter_pairs):
        for episode in layer:
            regularizer_value += abs(
                l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]]
            )  # Sum of abs differences between the episodes in all layers
    regularizer_value = np.exp(regularizer_value)
    print(regularizer_value)
    return regularizer_value


def custom_loss(lmbda, regularizer_value):
    def loss(y_true, y_pred):
        return F.cross_entropy(y_pred, y_true) + lmbda * regularizer_value

    return loss


def train(model, epochs, first_time, learning_rate=0.001):
    """
    Arguments:
        model: model to be trained
        epochs: number of epochs to be trained
        first_time: boolean indicating if it's the first time training
    Return:
        model: trained/fine-tuned model
        history: accuracies and losses
        weight_list_per_epoch: all weight tensors per epoch in a list
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(".", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    weight_list_per_epoch = []
    print("Training model")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
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
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(test_loader.dataset)
        val_accuracy = 100.0 * correct / len(test_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Save weights at the end of each epoch
        weight_list_per_epoch.append(
            [param.data.clone() for param in model.parameters()]
        )

    return model, history, weight_list_per_epoch

def logging(model, history, log_dict=None):
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
    initial_flops = count_model_params_flops(model, True, (1, 28,28))[1]
    print(f"Initial FLOPS: {initial_flops}")

    best_acc_index = history["val_accuracy"].index(max(history["val_accuracy"]))
    log_dict["train_loss"].append(history["loss"][best_acc_index])
    log_dict["train_acc"].append(history["accuracy"][best_acc_index])
    log_dict["val_loss"].append(history["val_loss"][best_acc_index])
    log_dict["val_acc"].append(history["val_accuracy"][best_acc_index])
    a, b = count_model_params_flops(model, True)
    log_dict["total_params"].append(a)
    log_dict["total_flops"].append(b)
    log_dict["filters_in_conv1"].append(model.conv1.weight.shape[0])
    log_dict["filters_in_conv2"].append(model.conv2.weight.shape[0])

    print("Validation accuracy ", max(history["val_accuracy"]))

    return log_dict

model = LeNet()


print("Model Initialized")

model, history, weight_list_per_epoch = train(model, 1, True)
log_dict = logging(model, history, None)

validation_accuracy = max(history["val_accuracy"])
max_val_acc = validation_accuracy
count = 0
all_models = list()
a, b = count_model_params_flops(model, False)
print(a, b)

print("Starting Pruning Process")

while validation_accuracy - max_val_acc >= -0.01:
    print("ITERATION {} ".format(count + 1))
    all_models.append(model)
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy

    if count < 1:
        optimize(model, weight_list_per_epoch, 1, 5, True)
        model = my_delete_filters(model, weight_list_per_epoch, 5, True)
        model, history, weight_list_per_epoch = train(model, 1, False)

    elif count < 2:
        optimize(model, weight_list_per_epoch, 1, 7, False)
        model = my_delete_filters(model, weight_list_per_epoch, 7, False)
        model, history, weight_list_per_epoch = train(model, 1, False)

    elif count < 3:
        optimize(model, weight_list_per_epoch, 1, 9, False)
        model = my_delete_filters(model, weight_list_per_epoch, 9, False)
        model, history, weight_list_per_epoch = train(model, 1, False)

    elif count < 4:
        optimize(model, weight_list_per_epoch, 1, 11, False)
        model = my_delete_filters(model, weight_list_per_epoch, 11, False)
        model, history, weight_list_per_epoch = train(model, 1, False)

    elif count < 5:
        optimize(model, weight_list_per_epoch, 1, 13, False)
        model = my_delete_filters(model, weight_list_per_epoch, 13, False)
        model, history, weight_list_per_epoch = train(model, 1, False)

    elif count < 10:
        optimize(model, weight_list_per_epoch, 1, 15, False)
        model = my_delete_filters(model, weight_list_per_epoch, 15, False)
        model, history, weight_list_per_epoch = train(model, 1, False)

    else:
        optimize(model, weight_list_per_epoch, 10, 35, False)
        model = my_delete_filters(model, weight_list_per_epoch, 35, False)
        model, history, weight_list_per_epoch = train(model, 10, False)

    a, b = count_model_params_flops(model, False)
    print(a, b)

    validation_accuracy = max(history["val_accuracy"])
    log_dict = logging(model, history, log_dict)
    print("VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(count + 1, validation_accuracy))
    count += 1

print(model)

model, history, weight_list_per_epoch = train(model, 60, False, learning_rate=0.001)
log_dict, max_val_acc, count, all_models = logging(model, history, log_dict)

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join('.', 'results', 'lenet5_2.csv'))