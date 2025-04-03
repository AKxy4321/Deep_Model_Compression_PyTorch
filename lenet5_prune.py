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

BATCH_SIZE = 1024
PRUNE_PER_LAYER = [2, 4]
MIN_FILTERS_PER_LAYER = [2, 3]


def LeNet5():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, bias=True),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(in_features=800, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
    )


# Add dataset = 0 so that config chooses MNIST dataset instead of CIFAR10
INPUT_SHAPE = config(BATCH_SIZE=BATCH_SIZE, dataset=0)

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
log_dict = logging(model, history, INPUT_SHAPE=INPUT_SHAPE)

max_val_acc = validation_accuracy

print(f"MAX VALIDATION ACCURACY = {max_val_acc}")
count = 0

print("STARTED PRUNING PROCESS")

while validation_accuracy - max_val_acc >= -1:
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

torch.save(model, os.path.join(os.getcwd(), "results", "lenet5_pruned.pt"))

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join(os.getcwd(), "results", "lenet5_mnist.csv"))
