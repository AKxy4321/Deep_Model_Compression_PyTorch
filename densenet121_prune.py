import os

import numpy as np
import pandas as pd
import torch
import torch_pruning as tp

from models.model_densenet121 import densenet121
from pruning_utils import (
    count_model_params_flops,
    delete_filters,
    device,
    logging,
)
from train_eval_optimise import config, evaluate, optimize, train

BATCH_SIZE = 128
INPUT_SHAPE = (BATCH_SIZE, 3, 32, 32)
NO_PRUNING_LIMIT = 8
PRUNE_PER_LAYER = [2] * 120


# Add dataset = 1 so that config chooses CIFAR10 dataset instead of MNIST
config(BATCH_SIZE=BATCH_SIZE, dataset=1)


model = densenet121(pretrained=False).to(device)
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
log_dict = logging(model, history, INPUT_SHAPE=INPUT_SHAPE)

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
    optimize(model, weight_list_per_epoch, 10, PRUNE_PER_LAYER)
    model, stop_flag = delete_filters(
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
    log_dict = logging(model, history, log_dict, INPUT_SHAPE=INPUT_SHAPE)
    print(
        "VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(
            count + 1, validation_accuracy
        )
    )

    # Break Loop if pruning indices == 0
    if stop_flag == 1:
        print("STOPPED PRUNING: NO MORE FILTERS TO PRUNE")
        break

    count += 1

print(model)

model, history, weight_list_per_epoch = train(model, 30, learning_rate=0.001)
log_dict = logging(model, history, log_dict, INPUT_SHAPE=INPUT_SHAPE)

torch.save(model, os.path.join(os.getcwd(), "results", "densenet121_pruned.pt"))

max_length = max(len(v) for v in log_dict.values())

# Pad shorter lists with NaN
log_dict = {k: v + [np.nan] * (max_length - len(v)) for k, v in log_dict.items()}

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join(os.getcwd(), "results", "densenet121_cifar10.csv"))
