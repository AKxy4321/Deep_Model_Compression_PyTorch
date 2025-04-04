import os
from itertools import combinations
from math import isqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torchprofile import profile_macs

dataset_path = os.path.join(os.getcwd(), "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_all_layers(module, layer_type, prefix="") -> list:
    layers = []
    for name, child in module.named_children():
        current_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, layer_type):
            layers.append(current_name)
        layers.extend(get_all_layers(child, layer_type, current_name))
    return layers


def get_all_conv_layers(model) -> list:
    return get_all_layers(model, nn.Conv2d)


def get_all_dense_layers(model) -> list:
    return get_all_layers(model, nn.Linear)


def get_weights_in_conv_layers(model) -> list:
    """
    Fetches the weights of all convolutional layers in the model.

    Args:
        model (nn.Module): The model to extract weights from.

    Returns:
        List of weight tensors from convolutional layers.
    """
    weights = []
    all_conv_layers = get_all_conv_layers(
        model
    )  # Gets layer names (int for Sequential, str otherwise)

    for layer_name in all_conv_layers:
        if isinstance(layer_name, int):  # For Sequential models
            weights.append(model[layer_name].weight.data.to(device))
        else:  # For general nn.Module models
            layer = dict(model.named_modules())[layer_name]
            weights.append(layer.weight.data.to(device))

    return weights


def get_cosine_sims_filters_per_epoch(weight_list_per_epoch: dict) -> dict:
    """
    Computes pairwise cosine similarity sums for filters of each layer.
    Assumes weight_list_per_epoch is a dict {layer_name: [tensor_per_epoch, ...]}.
    Returns a dict mapping layer names to a dict of filter pair similarity sums.
    """
    sorted_filter_pair_sum = {}
    num_filters_model = []
    for layer_name, epoch_weight_list in weight_list_per_epoch.items():
        num_filters = epoch_weight_list[0].shape[0]
        num_filters_model.append(num_filters)

        # Create a dict with filter pair keys (1-indexed in the key for clarity)
        filter_pair_similarities = {
            f"{i + 1}, {j + 1}": 0.0 for i, j in combinations(range(num_filters), 2)
        }

        for epoch_weights in epoch_weight_list:
            flattened_filters = epoch_weights.reshape(num_filters, -1)
            normed_filters = F.normalize(flattened_filters, p=2, dim=1)
            cosine_sim = torch.mm(normed_filters, normed_filters.T)

            # Sum similarities over epochs for each unique filter pair
            for i, j in combinations(range(num_filters), 2):
                filter_pair_similarities[f"{i + 1}, {j + 1}"] += cosine_sim[i, j].item()

        sorted_filter_pair_sum[layer_name] = dict(
            sorted(
                filter_pair_similarities.items(), key=lambda item: item[1], reverse=True
            )
        )

    total_filters = sum(num_filters_model)
    print(f"NUM FILTERS BEFORE PRUNING: {num_filters_model}")
    print(f"TOTAL NUMBER OF FILTERS {total_filters}")
    return sorted_filter_pair_sum, total_filters


def get_filter_pruning_indices(
    filter_pairs, l1_norms, num_filter_pairs_to_prune: int
) -> list:
    """
    Given a list of filter pairs, corresponding L1 norms, and the desired number of pairs,
    selects one filter from each pair to prune based on the lower L1 norm.

    Args:
        filter_pairs (list): List of pairs [filter_idx1, filter_idx2].
        l1_norms (Tensor or list): L1 norm values for the filters of a given layer.
        num_filter_pairs_to_prune (int): How many filter pairs to consider.

    Returns:
        list: Indices of filters selected for pruning.
    """
    num_filter_pairs_to_prune = min(len(filter_pairs), num_filter_pairs_to_prune)
    filter_pruning_indices = set()
    i = 0

    while len(filter_pruning_indices) < num_filter_pairs_to_prune and i < len(
        filter_pairs
    ):
        filter1, filter2 = filter_pairs[i]
        i += 1

        # Ensure we don't prune the same filter pair twice **within this layer**
        if filter1 in filter_pruning_indices or filter2 in filter_pruning_indices:
            continue  # Skip if either filter is already pruned

        if l1_norms[filter1] > l1_norms[filter2]:
            filter_pruning_indices.add(filter2)
        else:
            filter_pruning_indices.add(filter1)

    return sorted(list(filter_pruning_indices))


def find_pruning_indices(
    model,
    weight_list_per_epoch: dict,
    num_filter_pairs_to_prune_per_layer: list,
    min_filters_per_layer: list,
):
    """
    Finds pruning indices per layer based on cosine similarities between filters and L1 norms.

    Args:
        model: The model (used to get L1 norms via get_l1_norms).
        weight_list_per_epoch (dict): Dict mapping layer names to lists of weight tensors per epoch.
        num_filter_pairs_to_prune_per_layer (list): List of desired number of filter pairs to prune per layer.

    Returns:
        tuple: Two dictionaries: one mapping layer names to pruning indices, and the other mapping
               layer names to the list of filter pairs (each filter pair is a list of two indices).
    """
    sorted_filter_pair_sums, total_filters = get_cosine_sims_filters_per_epoch(
        weight_list_per_epoch
    )

    # Build a dict mapping layer names to their corresponding filter pairs (0-indexed)
    all_layer_filter_pairs = {}
    for layer_name, sorted_filter_pair_sum in sorted_filter_pair_sums.items():
        filter_pairs = []
        for key in sorted_filter_pair_sum.keys():
            # keys are in "i, j" format, where i and j are 1-indexed
            filter1, filter2 = map(int, key.split(","))
            filter_pairs.append([filter1 - 1, filter2 - 1])
        all_layer_filter_pairs[layer_name] = filter_pairs

    # Get L1 norms per layer (using updated get_l1_norms that uses named_modules()).
    l1_norm_matrix_dict = get_l1_norms(model)
    all_layer_pruning_indices = {}

    # Process layers in the order of sorted_filter_pair_sums keys
    for i, (layer_name, filter_pairs) in enumerate(all_layer_filter_pairs.items()):
        z = len(filter_pairs)
        total_filters_in_layer = (1 + isqrt(1 + 8 * z)) // 2
        min_filters = min_filters_per_layer[i]
        max_prunable = total_filters_in_layer - min_filters
        num_to_prune = min(num_filter_pairs_to_prune_per_layer[i], max_prunable)

        # Now use the layer_name key to access l1 norms
        pruning_indices = get_filter_pruning_indices(
            filter_pairs, l1_norm_matrix_dict[layer_name], num_to_prune
        )
        all_layer_pruning_indices[layer_name] = pruning_indices

    return all_layer_pruning_indices, all_layer_filter_pairs, total_filters


def get_l1_norms(model) -> dict:
    conv_layers = get_all_conv_layers(model)
    named_modules_dict = dict(model.named_modules())
    l1_norm_matrix_dict = {}

    for layer_name in conv_layers:
        layer = named_modules_dict.get(layer_name)
        if hasattr(layer, "weight") and layer.weight is not None:
            abs_weights = layer.weight.data.to(device).abs()
            l1_norm_matrix_dict[layer_name] = torch.sum(abs_weights, dim=(1, 2, 3))

    return l1_norm_matrix_dict


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_model_params_flops(model, input_shape):
    inputs = torch.randn(input_shape).to(device)
    macs = profile_macs(model, inputs)
    flops = 2 * macs

    params = count_parameters(model)
    return params, flops


class Get_Weights:
    def __init__(self):
        self.weight_list = []

    def on_epoch_end(self, epoch, model):
        if epoch == 0:
            all_conv_layers = get_all_conv_layers(model)
            for i in range(len(all_conv_layers)):
                self.weight_list.append([])

        for index, each_weight in enumerate(get_weights_in_conv_layers(model)):
            self.weight_list[index].append(each_weight)


def get_cosine_sims_filters(model) -> dict:
    conv_layers = get_all_conv_layers(model)  # e.g., ['features.0', 'features.3', ...]
    cosine_sums = {}
    named_modules_dict = dict(model.named_modules())

    for layer_name in conv_layers:
        layer = named_modules_dict[layer_name]
        weights = layer.weight.data.to(device)
        num_filters = weights.shape[0]

        filter_vectors = weights.view(num_filters, -1)
        normalized_filters = F.normalize(filter_vectors, p=2, dim=1)
        cosine_sims = torch.mm(normalized_filters, normalized_filters.T)
        cosine_sums_per_filter = cosine_sims - torch.eye(
            num_filters, device=weights.device
        )
        cosine_sums[layer_name] = [row.tolist() for row in cosine_sums_per_filter]

    return cosine_sums


def delete_filters(
    model,
    weight_list_per_epoch: dict,
    num_filter_pairs_to_prune_per_layer: list,
    min_filters_per_layer: list,
    input_shape=None,
    DG=None,
):
    if input_shape is None:
        print("Error: Input shape not defined")

    print("PRUNING STARTED")
    filter_pruning_indices, _, _ = find_pruning_indices(
        model=model,
        weight_list_per_epoch=weight_list_per_epoch,
        num_filter_pairs_to_prune_per_layer=num_filter_pairs_to_prune_per_layer,
        min_filters_per_layer=min_filters_per_layer,
    )
    all_conv_layers = get_all_conv_layers(model)
    layers = dict(model.named_modules())

    all_layers_empty = True

    num_filters_model = []
    for layer_name in all_conv_layers:
        layer = layers[layer_name]
        prune_indices = filter_pruning_indices[layer_name]

        if len(prune_indices) > 0:
            all_layers_empty = False

        if isinstance(layer, nn.Conv2d):
            group = DG.get_pruning_group(
                layer, tp.prune_conv_out_channels, idxs=prune_indices
            )
            if DG.check_pruning_group(group):  # Avoid over-pruning
                group.prune()
            else:
                print("invalid to prune more")

            num_filters_model.append(layer.out_channels)

    if all_layers_empty:
        return model, 1

    print(f"NUM FILTERS AFTER PRUNING: {num_filters_model}")
    return model, 0


def get_regularizer_value(
    model,
    weight_list_per_epoch: dict,
    num_filter_pairs_to_prune_per_layer: list,
):
    min_filters_per_layer = [0 for i in num_filter_pairs_to_prune_per_layer]
    _, filter_pairs_dict, total_filters = find_pruning_indices(
        model,
        weight_list_per_epoch,
        num_filter_pairs_to_prune_per_layer,
        min_filters_per_layer,
    )

    cosine_sims_dict = get_cosine_sims_filters(model)

    regularizer_value = 0
    for layer_name, layer in filter_pairs_dict.items():
        for episode in layer:
            regularizer_value += np.abs(
                np.sum(cosine_sims_dict[layer_name][episode[1]])
                - np.sum(cosine_sims_dict[layer_name][episode[0]])
            )

    regularizer_value = np.log1p(regularizer_value) * (
        regularizer_value / max(1, total_filters)
    )
    return regularizer_value


def custom_loss(lmbda: float = 0.1, regularizer_value: float = 0):
    def loss(y_pred, y_true):
        y_true = y_true.long()
        return F.cross_entropy(y_pred, y_true) + lmbda * regularizer_value

    return loss


def logging(model, history=None, log_dict=None, INPUT_SHAPE=None) -> dict:
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

    for layer_name in log_dict.keys():
        if layer_name.startswith("filters_in_"):
            conv_layer_name = layer_name.replace("filters_in_", "")
            conv_layer = dict(model.named_modules()).get(conv_layer_name)
            if conv_layer:
                log_dict[layer_name].append(conv_layer.out_channels)

    print("Validation accuracy ", max(history["val_accuracy"]))

    return log_dict
