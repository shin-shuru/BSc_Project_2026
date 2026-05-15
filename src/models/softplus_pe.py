from torch import nn

from src.models.relu_pe import PositionalEncoding


class SoftplusPEOccupancyNet(nn.Module):
    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        out_features: int = 1,
        num_frequencies: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
        pe_scale: float = 1.0,
        softplus_beta: float = 5.0,
        softplus_threshold: float = 20.0,
    ):
        """
        Softplus occupancy INR with fixed positional encoding.

        This is meant as a smooth MLP baseline for MAML. It keeps the same
        positional features as ReLU+PE, but replaces hard ReLU kinks with a
        differentiable activation that may behave better under second-order
        meta-learning.
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
            layers.append(
                nn.Softplus(
                    beta=float(softplus_beta),
                    threshold=float(softplus_threshold),
                )
            )
            last_features = hidden_features

        layers.append(nn.Linear(last_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.encoding(x))
