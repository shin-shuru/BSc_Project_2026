import json
from pathlib import Path

import numpy as np
import torch


def to_serializable(obj):
    """
    Recursively convert numpy / torch values into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()

    return obj


def save_case_outputs(
    case_dir: Path,
    model: torch.nn.Module,
    pred_mesh,
    loss_history: list[float],
    metrics: dict,
):
    """
    Save model checkpoint, predicted mesh, metrics, and loss history for one case.
    """
    case_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = case_dir / "model.pt"
    mesh_path = case_dir / "pred_mesh.stl"
    metrics_path = case_dir / "metrics.json"
    loss_path = case_dir / "loss_history.npy"

    torch.save(model.state_dict(), ckpt_path)
    pred_mesh.export(mesh_path)
    np.save(loss_path, np.array(loss_history, dtype=np.float32))

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(metrics), f, indent=2)