from torch import nn

from src.models.relu import ReLUOccupancyNet
from src.models.relu_pe import ReLUPEOccupancyNet
from src.models.siren import SirenOccupancyNet
from src.models.softplus_pe import SoftplusPEOccupancyNet


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

    if model_type in {"relu_pe", "relu+pe"}:
        return ReLUPEOccupancyNet(
            in_features=model_cfg.get("in_features", 3),
            hidden_features=model_cfg.get("hidden_features", 128),
            hidden_layers=model_cfg.get("hidden_layers", 3),
            out_features=model_cfg.get("out_features", 1),
            num_frequencies=model_cfg.get("num_frequencies", 6),
            include_input=model_cfg.get("include_input", True),
            log_sampling=model_cfg.get("log_sampling", True),
            pe_scale=model_cfg.get("pe_scale", 1.0),
            inplace_relu=model_cfg.get("inplace_relu", False),
        )

    if model_type in {"softplus_pe", "softplus+pe"}:
        return SoftplusPEOccupancyNet(
            in_features=model_cfg.get("in_features", 3),
            hidden_features=model_cfg.get("hidden_features", 128),
            hidden_layers=model_cfg.get("hidden_layers", 4),
            out_features=model_cfg.get("out_features", 1),
            num_frequencies=model_cfg.get("num_frequencies", 6),
            include_input=model_cfg.get("include_input", True),
            log_sampling=model_cfg.get("log_sampling", True),
            pe_scale=model_cfg.get("pe_scale", 1.0),
            softplus_beta=model_cfg.get("softplus_beta", 5.0),
            softplus_threshold=model_cfg.get("softplus_threshold", 20.0),
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
