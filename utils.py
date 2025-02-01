from itertools import combinations


import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn


def my_get_all_conv_layers(model, first_time):
    all_conv_layers = [
        i for i, layer in enumerate(model.layers) if isinstance(layer, nn.Conv2d)
    ]
    return all_conv_layers if first_time else all_conv_layers[1:]


def my_get_all_dense_layers(model):
    return [i for i, layer in enumerate(model.layers) if isinstance(layer, nn.Linear)]


def my_get_weights_in_conv_layers(model, first_time):
    weights = []
    all_conv_layers = my_get_all_conv_layers(model, first_time)

    for i in all_conv_layers:
        weights.append(list(model.children())[i].weight.data)

    return weights


def my_get_cosine_sims_filters_per_epoch(weight_list_per_epoch):
    num_layers = len(weight_list_per_epoch)
    num_filters = [
        torch.tensor(weight_list_per_epoch[i]).shape[-1] for i in range(num_layers)
    ]
    sorted_filter_pair_sum = [{} for _ in range(num_layers)]

    filter_pair_similarities = [
        {f"{i+1}, {j+1}": 0.0 for i, j in combinations(range(_), 2)}
        for _ in num_filters
    ]

    for layer_index in range(num_layers):
        for epochs in weight_list_per_epoch[layer_index]:
            epochs = torch.tensor(epochs, dtype=torch.float32)
            num_filter = num_filters[layer_index]
            flattened_filters = epochs.reshape(-1, num_filter).T

            normed_filters = F.normalize(flattened_filters, p=2, dim=1)
            cosine_sim = torch.mm(normed_filters, normed_filters.T)

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


def find_pruning_indices(model, weight_list_per_epoch, first_time, percentage):
    sorted_filter_pair_sums = my_get_cosine_sims_filters_per_epoch(
        weight_list_per_epoch
    )
    all_layer_filter_pairs = []

    for layer_index, sorted_filter_pair_sum in enumerate(sorted_filter_pair_sums):
        filter_pairs = []
        for key in sorted_filter_pair_sum.keys():
            filter1, filter2 = map(int, key.split(","))
            filter_pairs.append(
                [filter1 - 1, filter2 - 1]
            )  # Convert to zero-based index
        all_layer_filter_pairs.append(filter_pairs)

    l1_norm_matrix_list = l1_norms(model, first_time)
    all_layer_pruning_indices = []

    for layer_index, filter_pairs in enumerate(all_layer_filter_pairs):
        pruning_indices = my_get_filter_pruning_indices(
            filter_pairs, l1_norm_matrix_list[layer_index], percentage
        )
        all_layer_pruning_indices.append(pruning_indices)

    return all_layer_pruning_indices


def my_get_filter_pruning_indices(filter_pairs, l1_norms, prune_percentage):
    num_filter_pairs_to_prune = int(len(filter_pairs) * prune_percentage / 100 / 2)
    filter_pruning_indices = set()

    for i in range(num_filter_pairs_to_prune):
        filter1, filter2 = filter_pairs[i]
        if l1_norms[filter1] > l1_norms[filter2]:
            filter_pruning_indices.add(filter2)
        else:
            filter_pruning_indices.add(filter1)

    return list(filter_pruning_indices)


def l1_norms(model, first_time):
    conv_layers = my_get_all_conv_layers(model, first_time)
    l1_norms_list = []

    for layer_index in conv_layers:
        weights = list(model.children())[layer_index].weight.data
        layer_l1_norms = [
            torch.sum(torch.abs(weights[i])).item() for i in range(weights.shape[0])
        ]
        l1_norms_list.append(layer_l1_norms)

    return l1_norms_list


# def count_model_params_flops(model, first_time, input_shape):
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total_flops = 0

#     for i, layer in enumerate(model.children()):
#         if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
#             flops = conv_flops(layer)
#             print(
#                 i,
#                 layer.__class__.__name__,
#                 sum(p.numel() for p in layer.parameters() if p.requires_grad),
#                 flops,
#             )
#             total_flops += flops
#         elif isinstance(layer, nn.Linear):
#             flops = dense_flops(layer)
#             print(
#                 i,
#                 layer.__class__.__name__,
#                 sum(p.numel() for p in layer.parameters() if p.requires_grad),
#                 flops,
#             )
#             total_flops += flops

#     return total_params, int(total_flops)

# def dense_flops(layer):
#     output_channels = layer.out_features
#     input_channels = layer.in_features
#     return 2 * input_channels * output_channels

# def conv_flops(layer:nn.Conv2d):
#     conv_output_shape(layer.)
#     output_size = layer.shape[
#         2
#     ]  # Assuming output shape is (batch_size, channels, height, width)
#     kernel_shape = layer.weight.shape
#     return (
#         2
#         * (output_size**2)
#         * (kernel_shape[2] ** 2)
#         * kernel_shape[1]
#         * kernel_shape[0]
#     )
from torchprofile import profile_macs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_model_params_flops(model, first_time, input_shape):
    inputs = torch.randn(input_shape)
    macs = profile_macs(model, inputs)
    flops = 2 * macs

    params = count_parameters(model)
    return params, flops


class Get_Weights:
    def __init__(self, first_time):
        self.weight_list = []  # Using a list of list to store weight tensors per epoch
        self.first_time = first_time

    def on_epoch_end(self, epoch, model):
        if epoch == 0:
            all_conv_layers = my_get_all_conv_layers(model, self.first_time)
            for i in range(len(all_conv_layers)):
                self.weight_list.append(
                    []
                )  # appending empty lists for later appending weight tensors

        for index, each_weight in enumerate(
            my_get_weights_in_conv_layers(model, self.first_time)
        ):
            self.weight_list[index].append(each_weight)


def my_get_l1_norms_filters(model, first_time):
    """
    Arguments:
        model:

        first_time : type boolean
            first_time = True => model is not pruned
            first_time = False => model is pruned
        Return:
            l1_norms of all filters of every layer as a list
    """
    conv_layers = my_get_all_conv_layers(model, first_time)
    cosine_sums = list()
    for index, layer in enumerate(conv_layers):
        cosine_sums.append([])
        weights = layer.weight.data.cpu().numpy()
        num_filters = weights.shape[0]
        filter_vectors = [weights[i].flatten() for i in range(num_filters)]

        for i in range(num_filters):
            similarities = cosine_similarity([filter_vectors[i]], filter_vectors)[0]
            cosine_sum = np.sum(similarities) - 1
            cosine_sums[index].append(cosine_sum)

    return cosine_sums


def my_delete_filters(model, weight_list_per_epoch, percentage, first_time):
    filter_pruning_indices, _ = find_pruning_indices(
        model, weight_list_per_epoch, first_time, percentage
    )
    all_conv_layers = my_get_all_conv_layers(model, first_time)

    for index, (layer_name, layer) in enumerate(model.named_modules()):
        if layer in all_conv_layers:
            prune_indices = filter_pruning_indices[index]
            if isinstance(layer, nn.Conv2d):
                new_conv = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=layer.out_channels - len(prune_indices),
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=layer.bias is not None,
                )
                new_conv.weight.data = torch.cat(
                    [
                        layer.weight.data[i].unsqueeze(0)
                        for i in range(layer.out_channels)
                        if i not in prune_indices
                    ],
                    dim=0,
                )
                if layer.bias is not None:
                    new_conv.bias.data = torch.cat(
                        [
                            layer.bias.data[i].unsqueeze(0)
                            for i in range(layer.out_channels)
                            if i not in prune_indices
                        ],
                        dim=0,
                    )
                model._modules[layer_name] = new_conv

    return model
