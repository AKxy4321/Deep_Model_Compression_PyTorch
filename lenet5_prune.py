import os

import pandas as pd
import torch
import torch.nn as nn
import torch_pruning as tp

from pruning_utils import (
    count_model_params_flops,
    delete_filters,
    device,
    logging,
)
from train_eval_optimise import config, evaluate, optimize, train

BATCH_SIZE = 128
INPUT_SHAPE = (BATCH_SIZE, 1, 28, 28)
NO_PRUNING_LIMIT = 8
PRUNE_PER_LAYER = [2, 5]
MIN_FILTERS_PER_LAYER = [2, 3]


config(BATCH_SIZE=BATCH_SIZE)


def LeNet5():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=50 * 4 * 4, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
        nn.Softmax(dim=1),
    )


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


model = LeNet5().to(device)
model.load_state_dict(
    torch.load(os.path.join(os.getcwd(), "weights", "lenet5.pt"), weights_only=True)
)
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
total_params, total_flops = count_model_params_flops(model, INPUT_SHAPE)
print(total_params, total_flops)

print("STARTED PRUNING PROCESS")

iterations_without_pruning = 0
initial_parameters = sum(p.numel() for p in model.parameters())

while validation_accuracy - max_val_acc >= -1:
    current_parameters = sum(p.numel() for p in model.parameters())
    print("ITERATION {} ".format(count + 1))
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy

    print(f"MAX VALIDATION ACCURACY = {max_val_acc}")

    optimize(model, weight_list_per_epoch, 1, PRUNE_PER_LAYER)
    model = delete_filters(
        model=model,
        weight_list_per_epoch=weight_list_per_epoch,
        num_filter_pairs_to_prune_per_layer=PRUNE_PER_LAYER,
        min_filters_per_layer=MIN_FILTERS_PER_LAYER,
        input_shape=INPUT_SHAPE,
        DG=DG,
    )
    model, history, weight_list_per_epoch = train(model, 1)

    total_params, total_flops = count_model_params_flops(model, INPUT_SHAPE)
    print(total_params, total_flops)

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
log_df.to_csv(os.path.join(".", "results", "lenet5_2.csv"))
