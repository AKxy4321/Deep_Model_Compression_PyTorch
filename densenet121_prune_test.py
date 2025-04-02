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
MIN_FILTERS_PER_LAYER = [2] * 120

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

max_optimize_calls = 3  
optimize_calls = 0

while validation_accuracy - max_val_acc >= -1:
    print(f"ITERATION {count + 1}")

    if optimize_calls < max_optimize_calls:
        optimize(model, weight_list_per_epoch, 10, PRUNE_PER_LAYER)
        optimize_calls += 1
    else:
        print(f"Skipping further optimization after {max_optimize_calls} calls")
        break

    # Continue pruning and training
    model, stop_flag = delete_filters(
        model=model,
        weight_list_per_epoch=weight_list_per_epoch,
        num_filter_pairs_to_prune_per_layer=PRUNE_PER_LAYER,
        min_filters_per_layer=MIN_FILTERS_PER_LAYER,
        DG=DG,
        input_shape=INPUT_SHAPE,
    )
    model, history, weight_list_per_epoch = train(model, 10)

    validation_accuracy = max(history["val_accuracy"])
    if stop_flag == 1:
        print("STOPPED PRUNING: NO MORE FILTERS TO PRUNE")
        break

    count += 1

print(model)

model, history, weight_list_per_epoch = train(model, 30, learning_rate=0.001)
log_dict = logging(model, history, log_dict, INPUT_SHAPE=INPUT_SHAPE)

torch.save(model, os.path.join(os.getcwd(), "results", "densenet121_pruned.pt"))

log_df = pd.DataFrame(log_dict)
log_df.to_csv(os.path.join(os.getcwd(), "results", "densenet121_cifar10.csv"))
