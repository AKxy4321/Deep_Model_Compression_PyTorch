from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from lenet import LeNet
import torch.nn as nn
from tqdm import tqdm  
import pandas as pd
from utils import *
import torch
import os


INPUT_SHAPE = (1, 1, 28, 28)
def LeNet():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=50 * 4 * 4, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
        nn.Softmax(dim=1)
    )


def optimize(model, weight_list_per_epoch, epochs, percentage):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    regularizer_value = my_get_regularizer_value(model, weight_list_per_epoch, percentage)
    print("INITIAL REGULARIZER VALUE ", regularizer_value)
    criterion = custom_loss(lmbda=0.1, regularizer_value=regularizer_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        progress_bar = tqdm(train_loader, desc=f"Optimizing {epoch+1}/{epochs}", leave=False)

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

    print("FINAL REGULARIZER VALUE ", regularizer_value)
    return model, history


def train(model, epochs, learning_rate=0.001):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    conv_indices = my_get_all_conv_layers(model)
    weight_list_per_epoch = [[] for _ in conv_indices]

    print("Training model")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

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

    return model, history, weight_list_per_epoch


def logging(model, history, log_dict=None):
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

    # best_acc_index = history["val_accuracy"].index(max(history["val_accuracy"]))
    # log_dict["train_loss"].append(history["loss"][best_acc_index])
    # log_dict["train_acc"].append(history["accuracy"][best_acc_index])
    # log_dict["val_loss"].append(history["val_loss"][best_acc_index])
    # log_dict["val_acc"].append(history["val_accuracy"][best_acc_index])
    a, b = count_model_params_flops(model, INPUT_SHAPE)
    log_dict["total_params"].append(a)
    log_dict["total_flops"].append(b)
    if log_dict is not None:
        print(f"Current FLOPS: {b}, Current params : {a}")
    # log_dict["filters_in_conv1"].append(model[0].out_channels)
    # log_dict["filters_in_conv2"].append(model[3].out_channels)

    print("Validation accuracy ", max(history["val_accuracy"]))

    return log_dict

model = LeNet()


print("Model Initialized")

model, history, weight_list_per_epoch = train(model, 1)
log_dict = logging(model, history, None)

validation_accuracy = max(history["val_accuracy"])
max_val_acc = validation_accuracy
count = 0
all_models = list()
a, b = count_model_params_flops(model, INPUT_SHAPE)
print(a, b)

print("Starting Pruning Process")

while validation_accuracy - max_val_acc >= -0.01:
    print("ITERATION {} ".format(count + 1))
    all_models.append(model)
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy

    if count < 1:
        optimize(model, weight_list_per_epoch, 1, 5)
        model = my_delete_filters(model, weight_list_per_epoch, 5)
        print("Model after pruning:")
        print(model)
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 2:
        optimize(model, weight_list_per_epoch, 1, 7)
        model = my_delete_filters(model, weight_list_per_epoch, 7)
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 3:
        optimize(model, weight_list_per_epoch, 1, 9)
        model = my_delete_filters(model, weight_list_per_epoch, 9)
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 4:
        optimize(model, weight_list_per_epoch, 1, 11)
        model = my_delete_filters(model, weight_list_per_epoch, 11)
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 5:
        optimize(model, weight_list_per_epoch, 1, 13)
        model = my_delete_filters(model, weight_list_per_epoch, 13)
        model, history, weight_list_per_epoch = train(model, 1)

    elif count < 10:
        optimize(model, weight_list_per_epoch, 1, 15)
        model = my_delete_filters(model, weight_list_per_epoch, 15)
        model, history, weight_list_per_epoch = train(model, 1)

    else:
        optimize(model, weight_list_per_epoch, 10, 35)
        model = my_delete_filters(model, weight_list_per_epoch, 35)
        model, history, weight_list_per_epoch = train(model, 10)

    a, b = count_model_params_flops(model, INPUT_SHAPE)
    print(a, b)

    validation_accuracy = max(history["val_accuracy"])
    log_dict = logging(model, history, log_dict)
    print("VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(count + 1, validation_accuracy))
    count += 1

print(model)

model, history, weight_list_per_epoch = train(model, 60, learning_rate=0.001)
log_dict = logging(model, history, log_dict)

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join('.', 'results', 'lenet5_2.csv'))