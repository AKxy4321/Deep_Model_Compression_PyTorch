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

BATCH_SIZE = 512
PRUNE_PER_LAYER = [2] * 49
MIN_FILTERS_PER_LAYER = [2] * 49


# Add dataset = 1 so that config chooses CIFAR10 dataset instead of MNIST
INPUT_SHAPE = config(BATCH_SIZE=BATCH_SIZE, dataset=1)

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
log_dict = logging(model, history, INPUT_SHAPE=INPUT_SHAPE)

max_val_acc = validation_accuracy

print(f"MAX VALIDATION ACCURACY = {max_val_acc}")
count = 0

print("STARTED PRUNING PROCESS")

while validation_accuracy - max_val_acc >= -5:
    print("ITERATION {} ".format(count + 1))
    # if max_val_acc > validation_accuracy:
    #     max_val_acc = validation_accuracy

    #     print(f"MAX VALIDATION ACCURACY UPDATE TO = {max_val_acc}")

    model, history = optimize(model, weight_list_per_epoch, 10, PRUNE_PER_LAYER)
    model, stop_flag = delete_filters(
        model=model,
        weight_list_per_epoch=weight_list_per_epoch,
        num_filter_pairs_to_prune_per_layer=PRUNE_PER_LAYER,
        min_filters_per_layer=MIN_FILTERS_PER_LAYER,
        DG=DG,
        input_shape=INPUT_SHAPE,
    )

    model, history, weight_list_per_epoch = train(model, 10)

    a, b = count_model_params_flops(model, INPUT_SHAPE)
    print(a, b)

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

model, history, weight_list_per_epoch = train(model, 30)
log_dict = logging(model, history, log_dict, INPUT_SHAPE=INPUT_SHAPE)

torch.save(model, os.path.join(os.getcwd(), "results", "resnet50_pruned.pt"))

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join(os.getcwd(), "results", "resnet50_cifar10.csv"))
