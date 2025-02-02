from sklearn.metrics.pairwise import cosine_similarity
from torchvision import datasets, transforms
from torchprofile import profile_macs
from itertools import combinations
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


def my_get_all_conv_layers(model):
    all_conv_layers = []

    all_conv_layers = [
        i for i, layer in enumerate(model) if isinstance(layer, nn.Conv2d)
    ]

    return all_conv_layers


def my_get_all_dense_layers(model):
    return [i for i, layer in enumerate(model) if isinstance(layer, nn.Linear)]


def my_get_weights_in_conv_layers(model):
    weights = []
    all_conv_layers = my_get_all_conv_layers(model)

    for i in all_conv_layers:
        weights.append(list(model.children())[i].weight.data)

    return weights


def my_get_cosine_sims_filters_per_epoch(weight_list_per_epoch):
    num_layers = len(weight_list_per_epoch)
    num_filters = [
        len(weight_list_per_epoch[i][0]) for i in range(num_layers)
    ]  # function works

    print(num_filters)
    sorted_filter_pair_sum = [{} for _ in range(num_layers)]

    filter_pair_similarities = [
        {f"{i+1}, {j+1}": 0.0 for i, j in combinations(range(_), 2)}
        for _ in num_filters
    ]
    # print(f"Previous Filter Pairs {filter_pair_similarities}")

    for layer_index in range(num_layers):
        for epoch_weights in weight_list_per_epoch[
            layer_index
        ]:  # Each epoch's weights for this layer
            num_filter = num_filters[layer_index]
            # Reshape to [num_filters, -1] (each row is a flattened filter)
            flattened_filters = epoch_weights.reshape(num_filter, -1)

            normed_filters = F.normalize(flattened_filters, p=2, dim=1)
            cosine_sim = torch.mm(
                normed_filters, normed_filters.T
            )  # Pairwise cosine similarities

            for i, j in combinations(range(num_filter), 2):
                filter_pair_similarities[layer_index][f"{i+1}, {j+1}"] += cosine_sim[
                    i, j
                ].item()

    # Sort similarities
    for layer_index in range(num_layers):
        sorted_filter_pair_sum[layer_index] = dict(
            sorted(
                filter_pair_similarities[layer_index].items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    return sorted_filter_pair_sum


def find_pruning_indices(model, weight_list_per_epoch, percentage):
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

    l1_norm_matrix_list = l1_norms(model)
    all_layer_pruning_indices = []

    for layer_index, filter_pairs in enumerate(all_layer_filter_pairs):
        pruning_indices = my_get_filter_pruning_indices(
            filter_pairs, l1_norm_matrix_list[layer_index], percentage
        )
        all_layer_pruning_indices.append(pruning_indices)

    num_filter_pairs_to_prune = int(len(filter_pairs) * percentage / 100 / 2)
    return all_layer_pruning_indices, all_layer_filter_pairs[:num_filter_pairs_to_prune]


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


def l1_norms(model):
    conv_layers = my_get_all_conv_layers(model)
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
        self.weight_list = []  # Using a list of list to store weight tensors per epoch

    def on_epoch_end(self, epoch, model):
        if epoch == 0:
            all_conv_layers = my_get_all_conv_layers(model)
            for i in range(len(all_conv_layers)):
                self.weight_list.append(
                    []
                )  # appending empty lists for later appending weight tensors

        for index, each_weight in enumerate(my_get_weights_in_conv_layers(model)):
            self.weight_list[index].append(each_weight)


def my_get_cosine_sims_filters(model):
    """
    Arguments:
        model:

        first_time : type boolean
            first_time = True => model is not pruned
            first_time = False => model is pruned
        Return:
            l1_norms of all filters of every layer as a list
    """
    conv_layers = my_get_all_conv_layers(model)
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


def my_delete_filters(model, weight_list_per_epoch, percentage):
    filter_pruning_indices, _ = find_pruning_indices(
        model, weight_list_per_epoch, percentage
    )
    all_conv_layers = my_get_all_conv_layers(model)

    print("model\n", model)

    # Convert model to list format to replace layers
    layers = list(model.children())

    prev_num_out_channels = 1
    prev_remaining_out_channels = [0]
    for layer_index in range(len(all_conv_layers)):
        conv_idx = all_conv_layers[layer_index]
        layer = layers[conv_idx]
        prune_indices = filter_pruning_indices[layer_index]
        if isinstance(layer, nn.Conv2d):
            # New Conv2d with fewer filters
            remaining_filters = [
                i for i in range(layer.out_channels) if i not in prune_indices
            ]
            new_num_out_channels = len(remaining_filters)
            print("prev_out_channels", prev_num_out_channels)
            new_conv = nn.Conv2d(
                in_channels=prev_num_out_channels,
                out_channels=new_num_out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=layer.bias is not None,
            )

            # Copy only the remaining filters
            new_conv.weight.data = torch.index_select(
                layer.weight.data, 0, torch.tensor(remaining_filters)
            )
            new_conv.weight.data = torch.index_select(
                new_conv.weight.data, 1, torch.tensor(prev_remaining_out_channels)
            )
            if layer.bias is not None:
                new_conv.bias.data = torch.index_select(
                    layer.bias.data, 0, torch.tensor(remaining_filters)
                )

            # Replace the layer
            print(new_conv.weight.data.shape)
            layers[conv_idx] = new_conv

            prev_num_out_channels = new_num_out_channels
            prev_remaining_out_channels = remaining_filters

        # updating the linear layer immediately after the conv layer
    layer = layers[5]
    # Update the in_features of the first Linear layer if necessary
    new_in_features = (
        prev_num_out_channels * 4 * 4
    )  # Update based on the new conv output
    new_linear = nn.Linear(
        in_features=new_in_features,
        out_features=layer.out_features,
        bias=layer.bias is not None,
    )
    flattened_input_features_to_keep = get_flattened_indices(remaining_filters, 4, 4)
    previous_shape = layer.weight.data.shape
    current_shape = new_linear.weight.data.shape
    new_linear.weight.data = torch.index_select(
        layer.weight.data, 1, torch.tensor(flattened_input_features_to_keep)
    )
    layers[5] = new_linear

    # Reconstruct the model
    pruned_model = nn.Sequential(*layers)

    print("pruned model\n", pruned_model)

    input_shape = (128, 1, 28, 28)
    verify_shapes(pruned_model, input_shape)
    return pruned_model


def verify_shapes(model, input_shape):
    x = torch.randn(input_shape)
    for layer in model:
        x = layer(x)
        print(f"After {layer.__class__.__name__}: {x.shape}")


def get_flattened_indices(channels_to_keep, height, width):
    flattened_indices = []
    for channel_idx in channels_to_keep:
        for h in range(height):
            for w in range(width):
                flattened_index = channel_idx * height * width + h * width + w
                flattened_indices.append(flattened_index)
    return flattened_indices


def my_get_regularizer_value(model, weight_list_per_epoch, percentage):
    """
    Arguments:
        model: initial model
        weight_list_per_epoch: weight tensors at every epoch
        percentage: percentage of filter to be pruned
        first_time: type bool
    Return:
        regularizer_value
    """
    _, filter_pairs = find_pruning_indices(
        model, weight_list_per_epoch, percentage
    )
    l1_norms = my_get_cosine_sims_filters(model)
    regularizer_value = 0
    for layer_index, layer in enumerate(filter_pairs):
        for episode in layer:
            regularizer_value += abs(
                l1_norms[layer_index][episode[1]] - l1_norms[layer_index][episode[0]]
            )  # Sum of abs differences between the episodes in all layers
    regularizer_value = np.exp(regularizer_value)
    print(regularizer_value)
    return regularizer_value


def custom_loss(lmbda, regularizer_value):
    def loss(y_true, y_pred):
        return F.cross_entropy(y_pred, y_true) + lmbda * regularizer_value

    return loss