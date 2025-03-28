import os

import pandas as pd
import torch
import torch_pruning as tp

from models.model_resnet50 import resnet50
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
PRUNE_PER_LAYER = [2] * 49


config(BATCH_SIZE=BATCH_SIZE)


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
