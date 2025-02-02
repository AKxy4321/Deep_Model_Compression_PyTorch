import torch
import torch.nn as nn
import torch.nn.functional as F


def LeNet():
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=2, bias=False),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(in_features=50 * 4 * 4, out_features=500),
        nn.ReLU(),
        nn.Linear(in_features=500, out_features=10),
        nn.Softmax(dim=1)
    )
