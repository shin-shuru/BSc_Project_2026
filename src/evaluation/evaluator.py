import warnings
from pathlib import Path

import igl
import numpy as np
import trimesh

from src.evaluation.mesh_utils import MeshEvaluator


# Compatibility for igl
if not hasattr(igl, "fast_winding_number_for_meshes"):
    def fast_winding_number_for_meshes(V, F, Q):
        return igl.winding_number(V, F, Q)

    igl.fast_winding_number_for_meshes = fast_winding_number_for_meshes


def sanitize_metrics(obj):
    """
    Recursively replace NaN / inf values with safe Python values.
    """
    if isinstance(obj, dict):
        return {k: sanitize_metrics(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [sanitize_metrics(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(sanitize_metrics(v) for v in obj)

    if isinstance(obj, np.ndarray):
        return np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)

    if isinstance(obj, np.floating):
        return float(np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0))

    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj

    return obj


def evaluate_case(
    pred_mesh: trimesh.Trimesh,
    gt_mesh_path: Path,
    seed: int,
    n_cube: int = 256,
    n_pointcloud: int = 100_000,
    hash_resolution: int = 512,
) -> dict:
    """
    Evaluate predicted mesh against ground-truth mesh.
    """
    gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)

    mesh_evaluator = MeshEvaluator(
        N_pointcloud=n_pointcloud,
        N_cube=n_cube,
        min_max_range=[-0.5, 0.5],
        winding_number_threshold=0.5,
        hash_resolution=hash_resolution,
        verbose=False,
        random_seed=seed,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )
        metrics, _ = mesh_evaluator.eval_mesh(pred_mesh, gt_mesh)

    metrics = sanitize_metrics(metrics)

    metrics["pred_is_watertight"] = bool(pred_mesh.is_watertight)
    metrics["pred_is_volume"] = bool(pred_mesh.is_volume)
    metrics["num_vertices"] = int(len(pred_mesh.vertices))
    metrics["num_faces"] = int(len(pred_mesh.faces))

    return metrics