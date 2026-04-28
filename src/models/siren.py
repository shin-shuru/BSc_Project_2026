import math
import torch
from torch import nn

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, w0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SirenOccupancyNet(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=128,
        hidden_layers=5,
        out_features=1,
        first_w0=30.0,
        hidden_w0=30.0
    ):
        super().__init__()

        layers = [
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                w0=first_w0
            )
        ]

        for _ in range(hidden_layers):
            layers.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    w0=hidden_w0
                )
            )

        self.net = nn.Sequential(*layers)
        self.final_linear = nn.Linear(hidden_features, out_features)

        with torch.no_grad():
            bound = math.sqrt(6 / hidden_features) / hidden_w0
            self.final_linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        x = self.net(x)
        x = self.final_linear(x)   # raw logits
        return x