from sklearn.metrics.pairwise import cosine_similarity
from torchprofile import profile_macs
from itertools import combinations
import torch.nn.functional as F
import torch.optim as optim
import torch_pruning as tp
import torch.nn as nn
import numpy as np
import torch
import os


dataset_path = os.path.join(os.getcwd(), "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_all_conv_layers(model):
    all_conv_layers = []

    all_conv_layers = [
        i for i, layer in enumerate(model) if isinstance(layer, nn.Conv2d)
    ]

    return all_conv_layers


def get_all_dense_layers(model):
    return [i for i, layer in enumerate(model) if isinstance(layer, nn.Linear)]


def get_weights_in_conv_layers(model):
    weights = []
    all_conv_layers = get_all_conv_layers(model)

    for i in all_conv_layers:
        weights.append(list(model.children())[i].weight.data)

    return weights


def get_cosine_sims_filters_per_epoch(weight_list_per_epoch):
    num_layers = len(weight_list_per_epoch)
    num_filters = [len(weight_list_per_epoch[i][0]) for i in range(num_layers)]

    print(num_filters)
    sorted_filter_pair_sum = [{} for _ in range(num_layers)]

    filter_pair_similarities = [
        {f"{i+1}, {j+1}": 0.0 for i, j in combinations(range(_), 2)}
        for _ in num_filters
    ]

    for layer_index in range(num_layers):
        for epoch_weights in weight_list_per_epoch[layer_index]:
            num_filter = num_filters[layer_index]
            flattened_filters = epoch_weights.reshape(num_filter, -1)

            normed_filters = F.normalize(flattened_filters, p=2, dim=1)
            cosine_sim = torch.mm(
                normed_filters, normed_filters.T
            )  # Pairwise cosine similarities

            for i, j in combinations(range(num_filter), 2):
                filter_pair_similarities[layer_index][f"{i+1}, {j+1}"] += cosine_sim[
                    i, j
                ].item()

    for layer_index in range(num_layers):
        sorted_filter_pair_sum[layer_index] = dict(
            sorted(
                filter_pair_similarities[layer_index].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    return sorted_filter_pair_sum


def find_pruning_indices(
    model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer: list
):
    sorted_filter_pair_sums = get_cosine_sims_filters_per_epoch(weight_list_per_epoch)
    all_layer_filter_pairs = []

    for layer_index, sorted_filter_pair_sum in enumerate(sorted_filter_pair_sums):
        filter_pairs = []
        for key in sorted_filter_pair_sum.keys():
            filter1, filter2 = map(int, key.split(","))
            filter_pairs.append([filter1 - 1, filter2 - 1])
        all_layer_filter_pairs.append(filter_pairs)

    l1_norm_matrix_list = get_l1_norms(model)
    all_layer_pruning_indices = []

    for layer_index, filter_pairs in enumerate(all_layer_filter_pairs):
        z = len(filter_pairs)
        tot_filters_in_layer = int(round((1 + ((1 + 8 * z) ** 0.5)) / 2))
        pruning_indices = get_filter_pruning_indices(
            filter_pairs,
            l1_norm_matrix_list[layer_index],
            min(
                num_filter_pairs_to_prune_per_layer[layer_index],
                tot_filters_in_layer - 1,
            ),
        )
        all_layer_pruning_indices.append(pruning_indices)

    return all_layer_pruning_indices, all_layer_filter_pairs


def get_filter_pruning_indices(filter_pairs, l1_norms, num_filter_pairs_to_prune):
    filter_pruning_indices = set()

    for i in range(num_filter_pairs_to_prune):
        filter1, filter2 = filter_pairs[i]
        if l1_norms[filter1] > l1_norms[filter2]:
            filter_pruning_indices.add(filter2)
        else:
            filter_pruning_indices.add(filter1)

    return list(filter_pruning_indices)


def get_l1_norms(model):
    conv_layers = get_all_conv_layers(model)
    l1_norms_list = []

    for layer_index in conv_layers:
        weights = list(model.children())[layer_index].weight.data
        layer_l1_norms = [
            torch.sum(torch.abs(weights[i])).item() for i in range(weights.shape[0])
        ]
        l1_norms_list.append(layer_l1_norms)

    return l1_norms_list


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_model_params_flops(model, input_shape):
    inputs = torch.randn(input_shape)
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


def get_cosine_sims_filters(model):
    conv_layers = get_all_conv_layers(model)
    cosine_sums = list()
    for index, layer_idx in enumerate(conv_layers):
        layer = model[layer_idx]
        cosine_sums.append([])
        weights = layer.weight.data.cpu().numpy()
        num_filters = weights.shape[0]
        filter_vectors = [weights[i].flatten() for i in range(num_filters)]

        for i in range(num_filters):
            similarities = cosine_similarity([filter_vectors[i]], filter_vectors)[0]
            cosine_sum = np.sum(similarities) - 1
            cosine_sums[index].append(cosine_sum)

    return cosine_sums


def delete_filters(
    model,
    weight_list_per_epoch,
    num_filter_pairs_to_prune_per_layer: list,
    input_shape=None,
    DG=None,
):
    if input_shape is None:
        print("Error: Input shape not defined")

    filter_pruning_indices, _ = find_pruning_indices(
        model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer
    )
    all_conv_layers = get_all_conv_layers(model)

    layers = list(model.children())

    for layer_index in range(len(all_conv_layers)):
        conv_idx = all_conv_layers[layer_index]
        layer = layers[conv_idx]
        prune_indices = filter_pruning_indices[layer_index]
        if len(prune_indices) == 0:
            continue
        if isinstance(layer, nn.Conv2d):
            group = DG.get_pruning_group(
                layer, tp.prune_conv_out_channels, idxs=prune_indices
            )
            if DG.check_pruning_group(group):  # Avoid over-pruning
                print("pruning group")
                group.prune()
            else:
                print("invalid to prune more")

    verify_shapes(model, input_shape)
    return model


def verify_shapes(model, input_shape):
    x = torch.randn(input_shape)
    for layer in model:
        x = layer(x)
        # print(f"After {layer.__class__.__name__}: {x.shape}")


def get_flattened_indices(channels_to_keep, height, width):
    flattened_indices = []
    for channel_idx in channels_to_keep:
        for h in range(height):
            for w in range(width):
                flattened_index = channel_idx * height * width + h * width + w
                flattened_indices.append(flattened_index)
    return flattened_indices


def get_regularizer_value(
    model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer: list
):
    _, filter_pairs = find_pruning_indices(
        model, weight_list_per_epoch, num_filter_pairs_to_prune_per_layer
    )
    cosine_sims = get_cosine_sims_filters(model)
    regularizer_value = 0
    for layer_index, layer in enumerate(filter_pairs):
        for episode in layer:
            regularizer_value += abs(
                cosine_sims[layer_index][episode[1]]
                - cosine_sims[layer_index][episode[0]]
            )  # Sum of abs differences between the episodes in all layers
    regularizer_value = np.exp(regularizer_value)
    print(regularizer_value)
    return regularizer_value


def custom_loss(lmbda, regularizer_value):
    def loss(y_true, y_pred):
        return F.cross_entropy(y_pred, y_true) + lmbda * regularizer_value

    return loss
