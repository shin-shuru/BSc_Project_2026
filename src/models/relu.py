import math
import torch
from torch import nn

class ReLUOccupancyNet(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 5,
        out_features: int = 1,
    ):
        """
        ReLU-based occupancy INR.

        hidden_layers means the number of Linear + ReLU blocks.
        This matches the original ReLU code where num_layers=5 created
        five hidden Linear layers followed by one final output layer.
        """
        super().__init__()

        layers = []
        last_features = in_features

        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_features, hidden_features))
            layers.append(nn.ReLU(inplace=True))
            last_features = hidden_features

        layers.append(nn.Linear(last_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)