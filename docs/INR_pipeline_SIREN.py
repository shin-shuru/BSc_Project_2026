import os
import sys
import math
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import warnings
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import trimesh
import trimesh.repair
import igl
from skimage.measure import marching_cubes

# =========================
# Reproducibility
# =========================
def seed_all(seed=None, logger=None, verbose=False) -> tuple[torch._C.Generator, np.random.Generator]:
    """
    Set seed for reproducibility
    Ref: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is None:
        seed = 2024

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_generator = np.random.default_rng(seed=seed)
    torch_generator = torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if logger is None:
        print(f"Using random seed: {seed}")
    else:
        logger.info(f"Using random seed: {seed}")

    return torch_generator, numpy_generator


# =========================
# Paths
# =========================
WORKPLACE_ROOT = Path("/home/bsc18/workplace")
DATASET_ROOT = WORKPLACE_ROOT / "dataset" / "Totalsegmentator_dataset_v201_Liver" / "ok"
SAMPLED_ROOT = DATASET_ROOT / "sampled_20000"
NPY_ROOT = SAMPLED_ROOT / "npy" / "3D_Reconstruction"
MESH_ROOT = SAMPLED_ROOT / "mesh"
MESH_UTILS_DIR = WORKPLACE_ROOT / "khoa"

sys.path.append(str(MESH_UTILS_DIR))
import mesh_utils
from mesh_utils import MeshEvaluator
from scipy.spatial import KDTree

## Fix for F1-score
def fixed_OccNet_CD(pred_pointcloud, gt_pointcloud, pred_normals=None, gt_normals=None,
                    Fscore_thresholds=np.linspace(1./1000, 1, 1000)):
    """
    Same idea as mesh_utils.OccNet_CD, but compute F-score from per-point
    distances instead of from the mean distances.
    """
    # GT -> Pred
    completeness_dist, completeness_normals = mesh_utils.distance_p2p(
        points_src=gt_pointcloud,
        normals_src=gt_normals,
        points_tgt=pred_pointcloud,
        normals_tgt=pred_normals,
    )
    completeness = completeness_dist.mean()
    completeness2 = (completeness_dist ** 2).mean()
    completeness_normals = completeness_normals.mean()

    # Pred -> GT
    accuracy_dist, accuracy_normals = mesh_utils.distance_p2p(
        points_src=pred_pointcloud,
        normals_src=pred_normals,
        points_tgt=gt_pointcloud,
        normals_tgt=gt_normals,
    )
    accuracy = accuracy_dist.mean()
    accuracy2 = (accuracy_dist ** 2).mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer
    chamferL1 = 0.5 * (completeness + accuracy)
    chamferL2 = 0.5 * (completeness2 + accuracy2)

    # Normal consistency
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals

    # Correct F-score: use distance arrays, not scalar means
    recall = mesh_utils.get_threshold_percentage(completeness_dist, Fscore_thresholds)
    precision = mesh_utils.get_threshold_percentage(accuracy_dist, Fscore_thresholds)

    F_scores = []
    for p, r in zip(precision, recall):
        denom = p + r
        F_scores.append(0.0 if denom == 0 else 2 * p * r / denom)

    return {
        'completeness': completeness,
        'accuracy': accuracy,
        'chamfer-L1': chamferL1,
        'completeness2': completeness2,
        'accuracy2': accuracy2,
        'chamfer-L2': chamferL2,
        'normals completeness': completeness_normals,
        'normals accuracy': accuracy_normals,
        'normal consistency': normals_correctness,
        'f-scores': F_scores,
    }

# overwrite the buggy function from mesh_utils in memory only
mesh_utils.OccNet_CD = fixed_OccNet_CD


# =========================
# Compatibility for igl
# =========================
if not hasattr(igl, "fast_winding_number_for_meshes"):
    def fast_winding_number_for_meshes(V, F, Q):
        return igl.winding_number(V, F, Q)
    igl.fast_winding_number_for_meshes = fast_winding_number_for_meshes


# =========================
# Dataset
# =========================
class OccupancyDataset(Dataset):
    def __init__(self, coords: np.ndarray, occ: np.ndarray):
        self.coords = torch.from_numpy(coords).float()
        self.occ = torch.from_numpy(occ).float().unsqueeze(1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.occ[idx]


# =========================
# SIREN model
# =========================
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


# =========================
# Utilities
# =========================
def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_all_npy_files() -> list[Path]:
    return sorted(NPY_ROOT.rglob("*.npy"))


def build_mesh_lookup() -> dict[str, Path]:
    """
    Builds a lookup from mesh stem -> mesh path.
    You may need to adjust this if .npy and .stl naming formats differ.
    """
    mesh_files = sorted(MESH_ROOT.glob("*.stl"))
    return {p.stem: p for p in mesh_files}


def match_npy_to_mesh(npy_path: Path, mesh_lookup: dict[str, Path]) -> Path | None:
    """
    Adjust this function if your naming scheme differs.

    Current strategy:
    - try exact stem match
    - otherwise try substring containment
    """
    stem = npy_path.stem

    if stem in mesh_lookup:
        return mesh_lookup[stem]

    for mesh_stem, mesh_path in mesh_lookup.items():
        if stem in mesh_stem or mesh_stem in stem:
            return mesh_path

    return None


def load_case(npy_path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.load(npy_path)
    assert arr.ndim == 2 and arr.shape[1] == 4, f"Expected (N,4), got {arr.shape}"
    coords = arr[:, :3].astype(np.float32)
    occ = arr[:, 3].astype(np.float32)
    return coords, occ


def train_single_case(
    coords: np.ndarray,
    occ: np.ndarray,
    device: torch.device,
    seed: int,
    case_name: str = "",
    batch_size: int = 2048,
    lr: float = 5e-5,
    num_epochs: int = 500,
    hidden_features: int = 128,
    hidden_layers: int = 3,
    first_w0: float = 10.0,
    hidden_w0: float = 5.0,
):
    torch_gen, _ = seed_all(seed)

    dataset = OccupancyDataset(coords, occ)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch_gen,
    )

    model = SirenOccupancyNet(
        in_features=3,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=1,
        first_w0=first_w0,
        hidden_w0=hidden_w0,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loss_history = []

    bar_desc = f"Training {case_name}" if case_name else "Training"
    epoch_bar = tqdm(range(1, num_epochs + 1), desc=bar_desc, unit="epoch", leave=False)

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        for batch_coords, batch_occ in dataloader:
            batch_coords = batch_coords.to(device, non_blocking=True)
            batch_occ = batch_occ.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_coords)
            loss = criterion(logits, batch_occ)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_coords.size(0)

        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)
        epoch_bar.set_postfix(loss=f"{epoch_loss:.6f}")

    return model, loss_history


def reconstruct_mesh(
    model: nn.Module,
    device: torch.device,
    grid_res: int = 128,
    grid_min: float = -0.5,
    grid_max: float = 0.5,
    grid_batch_size: int = 200000,
) -> trimesh.Trimesh:
    """
    For 300+ livers, start with 128 or 256.
    1024 is too expensive for a batch sweep.
    """
    xs = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)
    ys = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)
    zs = np.linspace(grid_min, grid_max, grid_res, dtype=np.float32)

    grid_xyz = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    model.eval()
    pred_logits_list = []

    with torch.no_grad():
        for start in range(0, len(grid_xyz), grid_batch_size):
            end = start + grid_batch_size
            batch = torch.from_numpy(grid_xyz[start:end]).float().to(device)
            logits = model(batch)
            pred_logits_list.append(logits.squeeze(1).cpu().numpy())

    pred_logits = np.concatenate(pred_logits_list, axis=0)
    pred_probs = 1.0 / (1.0 + np.exp(-pred_logits))
    volume = pred_probs.reshape(grid_res, grid_res, grid_res)

    voxel_size = (grid_max - grid_min) / (grid_res - 1)
    volume_padded = np.pad(volume, pad_width=1, mode="constant", constant_values=0.0)

    verts, faces, normals, values = marching_cubes(
        volume=volume_padded,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts = verts - np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32)
    verts = verts + np.array([grid_min, grid_min, grid_min], dtype=np.float32)

    # removed 'post-processing step' for fair comparison
    # 
    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    return pred_mesh

def sanitize_metrics(obj): ### fix for F-score 0/0 = nan error
    """
    Recursively replace NaN / inf values with safe Python values.
    """
    if isinstance(obj, dict):
        return {k: sanitize_metrics(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_metrics(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_metrics(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    elif isinstance(obj, np.floating):
        return float(np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0))
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return obj
    else:
        return obj


def evaluate_case(
    pred_mesh: trimesh.Trimesh,
    gt_mesh_path: Path,
    seed: int,
) -> dict:
    gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)

    mesh_evaluator = MeshEvaluator(
        N_pointcloud=100_000,
        N_cube=256,
        min_max_range=[-0.5, 0.5],
        winding_number_threshold=0.5,
        hash_resolution=512,
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

def to_serializable(obj):
    """
    Recursively convert numpy / torch types into native Python types
    so they can be written with json.dump().
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    else:
        return obj

def save_case_outputs(
    case_dir: Path,
    model: nn.Module,
    pred_mesh: trimesh.Trimesh,
    loss_history: list[float],
    metrics: dict,
):
    case_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = case_dir / "model.pt"
    mesh_path = case_dir / "pred_mesh.stl"
    metrics_path = case_dir / "metrics.json"
    loss_path = case_dir / "loss_history.npy"

    torch.save(model.state_dict(), ckpt_path)
    pred_mesh.export(mesh_path)
    np.save(loss_path, np.array(loss_history, dtype=np.float32))

    with open(metrics_path, "w") as f:
        json.dump(to_serializable(metrics), f, indent=2)


def run_one_case(
    npy_path: Path,
    gt_mesh_path: Path,
    output_root: Path,
    device: torch.device,
    seed: int,
    batch_size: int,
    lr: float,
    num_epochs: int,
    grid_res: int,
):
    case_name = npy_path.stem
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    coords, occ = load_case(npy_path)

    model, loss_history = train_single_case(
        coords=coords,
        occ=occ,
        device=device,
        seed=seed,
        case_name=case_name,
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
    )

    pred_mesh = reconstruct_mesh(
        model=model,
        device=device,
        grid_res=grid_res,
    )

    metrics = evaluate_case(
        pred_mesh=pred_mesh,
        gt_mesh_path=gt_mesh_path,
        seed=seed,
    )

    metrics["case_name"] = case_name
    metrics["seed"] = seed
    metrics["npy_path"] = str(npy_path)
    metrics["gt_mesh_path"] = str(gt_mesh_path)
    metrics["final_loss"] = float(loss_history[-1])

    save_case_outputs(
        case_dir=case_dir,
        model=model,
        pred_mesh=pred_mesh,
        loss_history=loss_history,
        metrics=metrics,
    )

    return metrics


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="/home/bsc18/workplace/Jongwon/outputs/batch_inr_siren") ## edit
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N cases")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    all_npy_files = find_all_npy_files()
    mesh_lookup = build_mesh_lookup()

    print(f"Found {len(all_npy_files)} npy files")

    summary_rows = []

    total_cases = len(all_npy_files) if args.limit is None else min(len(all_npy_files), args.limit)

    for i, npy_path in enumerate(tqdm(all_npy_files[:total_cases], desc="Cases", unit="case")):
        if args.limit is not None and i >= args.limit:
            break

        gt_mesh_path = match_npy_to_mesh(npy_path, mesh_lookup)
        if gt_mesh_path is None:
            print(f"[SKIP] No matching mesh for {npy_path.name}")
            summary_rows.append({
                "case_name": npy_path.stem,
                "status": "missing_gt_mesh",
                "npy_path": str(npy_path),
            })
            continue

        case_seed = args.seed

        print(f"\n[{i+1}/{total_cases}] {npy_path.name}")

        try:
            metrics = run_one_case(
                npy_path=npy_path,
                gt_mesh_path=gt_mesh_path,
                output_root=output_root,
                device=device,
                seed=case_seed,
                batch_size=args.batch_size,
                lr=args.lr,
                num_epochs=args.epochs,
                grid_res=args.grid_res,
            )
            metrics["status"] = "ok"
            summary_rows.append(to_serializable(metrics))

        except Exception as e:
            print(f"[ERROR] {npy_path.name}: {e}")
            summary_rows.append({
                "case_name": npy_path.stem,
                "status": "error",
                "error": str(e),
                "npy_path": str(npy_path),
                "gt_mesh_path": str(gt_mesh_path),
                "seed": case_seed,
            })

        pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)

    print("\nDone.")
    print("Summary saved to:", output_root / "_summary.csv")


if __name__ == "__main__":
    main()
