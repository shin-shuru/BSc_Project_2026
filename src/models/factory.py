from torch import nn

from src.models.relu import ReLUOccupancyNet
from src.models.siren import SirenOccupancyNet


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    model_type = model_cfg["type"].lower()

    if model_type == "relu":
        return ReLUOccupancyNet(
            in_features=model_cfg.get("in_features", 3),
            hidden_features=model_cfg.get("hidden_features", 128),
            hidden_layers=model_cfg.get("hidden_layers", 5),
            out_features=model_cfg.get("out_features", 1),
        )

    if model_type == "siren":
        return SirenOccupancyNet(
            in_features=model_cfg.get("in_features", 3),
            hidden_features=model_cfg.get("hidden_features", 128),
            hidden_layers=model_cfg.get("hidden_layers", 3),
            out_features=model_cfg.get("out_features", 1),
            first_w0=model_cfg.get("first_w0", 15.0),
            hidden_w0=model_cfg.get("hidden_w0", 15.0),
        )

    raise ValueError(f"Unknown model type: {model_type}")