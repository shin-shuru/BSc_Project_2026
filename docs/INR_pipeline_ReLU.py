import os
import sys
import json
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import trimesh
import igl
from skimage.measure import marching_cubes


# =========================
# Reproducibility
# =========================
def seed_all(seed=None, logger=None, verbose=False) -> tuple[torch._C.Generator, np.random.Generator]:
    """
    Set seed for reproducibility.
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
import mesh_utils  # noqa: E402
from mesh_utils import MeshEvaluator  # noqa: E402


# =========================
# Fix for F-score
# =========================
def fixed_OccNet_CD(
    pred_pointcloud,
    gt_pointcloud,
    pred_normals=None,
    gt_normals=None,
    Fscore_thresholds=np.linspace(1.0 / 1000, 1, 1000),
):
    """
    Same idea as mesh_utils.OccNet_CD, but computes F-score from per-point
    distances instead of scalar mean distances.
    """
    completeness_dist, completeness_normals = mesh_utils.distance_p2p(
        points_src=gt_pointcloud,
        normals_src=gt_normals,
        points_tgt=pred_pointcloud,
        normals_tgt=pred_normals,
    )
    completeness = completeness_dist.mean()
    completeness2 = (completeness_dist ** 2).mean()
    completeness_normals = completeness_normals.mean()

    accuracy_dist, accuracy_normals = mesh_utils.distance_p2p(
        points_src=pred_pointcloud,
        normals_src=pred_normals,
        points_tgt=gt_pointcloud,
        normals_tgt=gt_normals,
    )
    accuracy = accuracy_dist.mean()
    accuracy2 = (accuracy_dist ** 2).mean()
    accuracy_normals = accuracy_normals.mean()

    chamferL1 = 0.5 * (completeness + accuracy)
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals

    recall = mesh_utils.get_threshold_percentage(completeness_dist, Fscore_thresholds)
    precision = mesh_utils.get_threshold_percentage(accuracy_dist, Fscore_thresholds)

    F_scores = []
    for p, r in zip(precision, recall):
        denom = p + r
        F_scores.append(0.0 if denom == 0 else 2 * p * r / denom)

    return {
        "completeness": completeness,
        "accuracy": accuracy,
        "chamfer-L1": chamferL1,
        "completeness2": completeness2,
        "accuracy2": accuracy2,
        "chamfer-L2": chamferL2,
        "normals completeness": completeness_normals,
        "normals accuracy": accuracy_normals,
        "normal consistency": normals_correctness,
        "f-scores": F_scores,
    }


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
# ReLU model
# =========================
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


# =========================
# Utilities
# =========================
def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_case_id(name: str) -> str:
    stem = Path(name).stem
    suffixes = ["_Segment_1", "_segment_1", "_mesh", "_predicted_mesh"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_all_npy_files(npy_root: Path = NPY_ROOT) -> list[Path]:
    return sorted(npy_root.rglob("*.npy"))


def build_mesh_lookup(mesh_root: Path = MESH_ROOT) -> dict[str, Path]:
    """
    Build a lookup from normalized mesh stem to mesh path.
    """
    mesh_files = sorted(mesh_root.glob("*.stl"))
    return {normalize_case_id(p.name): p for p in mesh_files}


def match_npy_to_mesh(npy_path: Path, mesh_lookup: dict[str, Path]) -> Path | None:
    """
    Match .npy file to .stl mesh using normalized case IDs.
    """
    case_id = normalize_case_id(npy_path.name)

    if case_id in mesh_lookup:
        return mesh_lookup[case_id]

    candidates = [
        mesh_path
        for mesh_stem, mesh_path in mesh_lookup.items()
        if mesh_stem.startswith(case_id) or case_id.startswith(mesh_stem)
    ]

    if len(candidates) == 1:
        return candidates[0]

    return None


def select_npy_files_by_split(
    npy_files: list[Path],
    split: str,
    train_split: Path,
    val_split: Path,
    test_split: Path,
    case_ids: list[str] | None = None,
) -> list[Path]:
    if case_ids:
        allowed = {normalize_case_id(x) for x in case_ids}
        return [p for p in npy_files if normalize_case_id(p.name) in allowed]

    if split == "all":
        return npy_files

    split_map = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }
    allowed = {normalize_case_id(x) for x in read_split_file(split_map[split])}
    return [p for p in npy_files if normalize_case_id(p.name) in allowed]


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
    lr: float = 5e-4,
    num_epochs: int = 500,
    hidden_features: int = 256,
    hidden_layers: int = 5,
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

    model = ReLUOccupancyNet(
        in_features=3,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=1,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
    grid_res: int = 256,
    grid_min: float = -0.5,
    grid_max: float = 0.5,
    grid_batch_size: int = 200000,
) -> trimesh.Trimesh:
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

    verts, faces, _, _ = marching_cubes(
        volume=volume_padded,
        level=0.5,
        spacing=(voxel_size, voxel_size, voxel_size),
    )

    verts = verts - np.array([voxel_size, voxel_size, voxel_size], dtype=np.float32)
    verts = verts + np.array([grid_min, grid_min, grid_min], dtype=np.float32)

    # Same as the SIREN style file: no extra repair step here for fair comparison.
    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    return pred_mesh


def sanitize_metrics(obj):
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
    n_cube: int = 256,
) -> dict:
    gt_mesh = trimesh.load_mesh(gt_mesh_path, process=False)

    mesh_evaluator = MeshEvaluator(
        N_pointcloud=100_000,
        N_cube=n_cube,
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

    with open(metrics_path, "w", encoding="utf-8") as f:
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
    grid_batch_size: int,
    hidden_features: int,
    hidden_layers: int,
    n_cube: int,
):
    case_name = normalize_case_id(npy_path.name)
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
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
    )

    pred_mesh = reconstruct_mesh(
        model=model,
        device=device,
        grid_res=grid_res,
        grid_batch_size=grid_batch_size,
    )

    metrics = evaluate_case(
        pred_mesh=pred_mesh,
        gt_mesh_path=gt_mesh_path,
        seed=seed,
        n_cube=n_cube,
    )

    metrics["case_name"] = case_name
    metrics["seed"] = seed
    metrics["npy_path"] = str(npy_path)
    metrics["gt_mesh_path"] = str(gt_mesh_path)
    metrics["final_loss"] = float(loss_history[-1])
    metrics["model_type"] = "ReLUOccupancyNet"
    metrics["hidden_features"] = hidden_features
    metrics["hidden_layers"] = hidden_layers
    metrics["lr"] = lr
    metrics["epochs"] = num_epochs
    metrics["batch_size"] = batch_size
    metrics["grid_res"] = grid_res

    save_case_outputs(
        case_dir=case_dir,
        model=model,
        pred_mesh=pred_mesh,
        loss_history=loss_history,
        metrics=metrics,
    )

    return metrics


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Batch ReLU INR training + mesh evaluation output format")

    parser.add_argument("--npy_root", type=str, default=str(NPY_ROOT))
    parser.add_argument("--mesh_root", type=str, default=str(MESH_ROOT))
    parser.add_argument("--output_root", type=str, default="/home/bsc18/workplace/Jaehyeong/outputs/batch_inr_relu_2")

    parser.add_argument("--train_split", type=str, default=str(DATASET_ROOT / "train.txt"))
    parser.add_argument("--val_split", type=str, default=str(DATASET_ROOT / "val.txt"))
    parser.add_argument("--test_split", type=str, default=str(DATASET_ROOT / "test.txt"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--case_ids", nargs="*", default=None)

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--hidden_features", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=5)

    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--grid_batch_size", type=int, default=200000)
    parser.add_argument("--n_cube", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Only run first N matched cases")

    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    device = get_device(args.device)
    print("Using device:", device)

    npy_root = Path(args.npy_root)
    mesh_root = Path(args.mesh_root)
    train_split = Path(args.train_split)
    val_split = Path(args.val_split)
    test_split = Path(args.test_split)

    all_npy_files = find_all_npy_files(npy_root)
    selected_npy_files = select_npy_files_by_split(
        npy_files=all_npy_files,
        split=args.split,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        case_ids=args.case_ids,
    )

    if args.limit is not None:
        selected_npy_files = selected_npy_files[: args.limit]

    mesh_lookup = build_mesh_lookup(mesh_root)

    print(f"Found {len(all_npy_files)} total npy files")
    print(f"Running {len(selected_npy_files)} npy files from split='{args.split}'")

    summary_rows = []
    total_cases = len(selected_npy_files)

    for i, npy_path in enumerate(tqdm(selected_npy_files, desc="Cases", unit="case")):
        gt_mesh_path = match_npy_to_mesh(npy_path, mesh_lookup)
        case_name = normalize_case_id(npy_path.name)

        if gt_mesh_path is None:
            print(f"[SKIP] No matching mesh for {npy_path.name}")
            summary_rows.append({
                "case_name": case_name,
                "status": "missing_gt_mesh",
                "npy_path": str(npy_path),
            })
            pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)
            continue

        case_seed = args.seed
        print(f"\n[{i + 1}/{total_cases}] {npy_path.name}")

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
                grid_batch_size=args.grid_batch_size,
                hidden_features=args.hidden_features,
                hidden_layers=args.hidden_layers,
                n_cube=args.n_cube,
            )
            metrics["status"] = "ok"
            summary_rows.append(to_serializable(metrics))

        except Exception as e:
            print(f"[ERROR] {npy_path.name}: {e}")
            summary_rows.append({
                "case_name": case_name,
                "status": "error",
                "error": str(e),
                "npy_path": str(npy_path),
                "gt_mesh_path": str(gt_mesh_path),
                "seed": case_seed,
            })

        pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)

    print("\nDone.")
    print("Summary saved to:", output_root / "summary.csv")


if __name__ == "__main__":
    main()
