import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes


def sigmoid_stable(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid to avoid overflow warnings from np.exp.
    """
    logits = np.asarray(logits)

    probs = np.empty_like(logits, dtype=np.float32)

    positive = logits >= 0
    negative = ~positive

    probs[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))

    exp_x = np.exp(logits[negative])
    probs[negative] = exp_x / (1.0 + exp_x)

    return probs


def reconstruct_mesh(
    model: torch.nn.Module,
    device: torch.device,
    grid_res: int = 256,
    grid_min: float = -0.5,
    grid_max: float = 0.5,
    grid_batch_size: int = 200_000,
) -> trimesh.Trimesh:
    """
    Reconstruct mesh from trained occupancy INR using marching cubes.

    No repair / post-processing is applied except trimesh object construction.
    """
    xs = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)
    ys = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)
    zs = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)

    grid_xyz = np.stack(
        np.meshgrid(xs, ys, zs, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)

    model.eval()
    pred_logits_list = []

    with torch.no_grad():
        for start in range(0, len(grid_xyz), grid_batch_size):
            end = start + grid_batch_size

            batch = torch.from_numpy(grid_xyz[start:end]).float().to(device)
            logits = model(batch)

            pred_logits_list.append(logits.squeeze(1).cpu().numpy())

    pred_logits = np.concatenate(pred_logits_list, axis=0)

    pred_probs = sigmoid_stable(pred_logits)
    volume = pred_probs.reshape(grid_res, grid_res, grid_res)

    voxel_size = (grid_max - grid_min) / (grid_res - 1)

    volume_padded = np.pad(
        volume,
        pad_width=1,
        mode="constant",
        constant_values=0.0,
    )

    verts, faces, _, _ = marching_cubes(
        volume=volume_padded,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts = verts - np.array(
        [voxel_size, voxel_size, voxel_size],
        dtype=np.float32,
    )

    verts = verts + np.array(
        [grid_min, grid_min, grid_min],
        dtype=np.float32,
    )

    pred_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=True,
    )

    return pred_mesh