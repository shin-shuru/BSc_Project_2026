import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
        pe_scale: float = 1.0,
    ):
        """
        Fixed sinusoidal positional encoding for coordinate MLPs.

        For input x, this returns:
            [x, sin(pi * f_i * x), cos(pi * f_i * x)]
        where f_i are powers of two by default.
        """
        super().__init__()

        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive.")

        self.in_features = int(in_features)
        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.log_sampling = bool(log_sampling)
        self.pe_scale = float(pe_scale)

        if self.log_sampling:
            freq_bands = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)
        else:
            freq_bands = torch.linspace(
                1.0,
                2.0 ** (self.num_frequencies - 1),
                steps=self.num_frequencies,
                dtype=torch.float32,
            )

        self.register_buffer("freq_bands", freq_bands * self.pe_scale)

    @property
    def out_features(self) -> int:
        raw_features = self.in_features if self.include_input else 0
        encoded_features = 2 * self.num_frequencies * self.in_features
        return raw_features + encoded_features

    def forward(self, x):
        encoded = []

        if self.include_input:
            encoded.append(x)

        scaled = math.pi * x.unsqueeze(-2) * self.freq_bands.view(1, -1, 1)
        encoded.append(torch.sin(scaled).flatten(start_dim=-2))
        encoded.append(torch.cos(scaled).flatten(start_dim=-2))

        return torch.cat(encoded, dim=-1)


class ReLUPEOccupancyNet(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        out_features: int = 1,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
        pe_scale: float = 1.0,
        inplace_relu: bool = False,
    ):
        """
        ReLU-based occupancy INR with fixed positional encoding.

        This is intentionally separate from ReLUOccupancyNet so the raw ReLU
        baseline remains unchanged.
        """
        super().__init__()

        self.encoding = PositionalEncoding(
            in_features=in_features,
            num_frequencies=num_frequencies,
            include_input=include_input,
            log_sampling=log_sampling,
            pe_scale=pe_scale,
        )

        layers = []
        last_features = self.encoding.out_features

        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_features, hidden_features))
            layers.append(nn.ReLU(inplace=inplace_relu))
            last_features = hidden_features

        layers.append(nn.Linear(last_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.encoding(x))
