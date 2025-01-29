from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

def my_get_all_conv_layers(model, first_time):
    all_conv_layers = [i for i, layer in enumerate(model.layers) if isinstance(layer, nn.Conv2d)]
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
    num_filters = [torch.tensor(weight_list_per_epoch[i]).shape[-1] for i in range(num_layers)]
    sorted_filter_pair_sum = [{} for _ in range(num_layers)]
    
    filter_pair_similarities = [{f'{i+1}, {j+1}': 0.0 for i, j in combinations(range(_), 2)} for _ in num_filters]
    
    for layer_index in range(num_layers):
        for epochs in weight_list_per_epoch[layer_index]:
            epochs = torch.tensor(epochs, dtype=torch.float32)
            num_filter = num_filters[layer_index]
            flattened_filters = epochs.reshape(-1, num_filter).T  
            
            normed_filters = F.normalize(flattened_filters, p=2, dim=1)
            cosine_sim = torch.mm(normed_filters, normed_filters.T)
            
            for (i, j) in combinations(range(num_filter), 2):
                filter_pair_similarities[layer_index][f'{i+1}, {j+1}'] += cosine_sim[i, j].item()
    
    for layer_index in range(num_layers):  
        sorted_filter_pair_sum[layer_index] = dict(sorted(filter_pair_similarities[layer_index].items(), key=lambda item: item[1], reverse=True))
    
    return sorted_filter_pair_sum

def find_pruning_indices(model, weight_list_per_epoch, first_time, percentage):
    sorted_filter_pair_sums = my_get_cosine_sims_filters_per_epoch(weight_list_per_epoch)
    all_layer_filter_pairs = []

    for layer_index, sorted_filter_pair_sum in enumerate(sorted_filter_pair_sums):
        filter_pairs = []
        for key in sorted_filter_pair_sum.keys():
            filter1, filter2 = map(int, key.split(','))
            filter_pairs.append([filter1 - 1, filter2 - 1])  # Convert to zero-based index
        all_layer_filter_pairs.append(filter_pairs)
    
    l1_norm_matrix_list = l1_norms(model, first_time)
    all_layer_pruning_indices = []
    
    for layer_index, filter_pairs in enumerate(all_layer_filter_pairs):
        pruning_indices = my_get_filter_pruning_indices(filter_pairs, l1_norm_matrix_list[layer_index], percentage)
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
        layer_l1_norms = [torch.sum(torch.abs(weights[i])).item() for i in range(weights.shape[0])]
        l1_norms_list.append(layer_l1_norms)
    
    return l1_norms_list

def conv_flops(layer, input_shape):
    output_shape = layer(torch.zeros(1, *input_shape)).shape
    kernel_ops = torch.prod(torch.tensor(layer.kernel_size))
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    output_elements = torch.prod(torch.tensor(output_shape[2:]))
    flops = out_channels * output_elements * (in_channels * kernel_ops + 1)  
    return flops.item()

def dense_flops(layer):
    return 2 * layer.in_features * layer.out_features

def count_model_params_flops(model, first_time, input_shape):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_flops = 0
    
    for i, layer in enumerate(model.children()):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            flops = conv_flops(layer, input_shape)
            print(i, layer.__class__.__name__, sum(p.numel() for p in layer.parameters() if p.requires_grad), flops)
            total_flops += flops
        elif isinstance(layer, nn.Linear):
            flops = dense_flops(layer)
            print(i, layer.__class__.__name__, sum(p.numel() for p in layer.parameters() if p.requires_grad), flops)
            total_flops += flops
    
    return total_params, int(total_flops)
