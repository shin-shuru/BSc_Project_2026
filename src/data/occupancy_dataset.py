from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class OccupancyDataset(Dataset):
    def __init__(self, coords: np.ndarray, occ: np.ndarray):
        self.coords = torch.from_numpy(coords).float()
        self.occ = torch.from_numpy(occ).float().unsqueeze(1)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx], self.occ[idx]


def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_case_id(name: str) -> str:
    """
    Normalize file names so .npy and .stl names can be matched.
    """
    stem = Path(name).stem

    suffixes = [
        "_Segment_1",
        "_segment_1",
        "_mesh",
        "_predicted_mesh",
    ]

    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def find_all_npy_files(npy_root: Path) -> list[Path]:
    return sorted(npy_root.rglob("*.npy"))


def build_mesh_lookup(mesh_root: Path) -> dict[str, Path]:
    """
    Build lookup from normalized mesh stem to mesh path.
    """
    mesh_files = sorted(mesh_root.glob("*.stl"))
    return {normalize_case_id(p.name): p for p in mesh_files}


def match_npy_to_mesh(
    npy_path: Path,
    mesh_lookup: dict[str, Path],
) -> Path | None:
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
    """
    Select npy files by split or explicit case IDs.
    """
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

    assert arr.ndim == 2 and arr.shape[1] == 4, (
        f"Expected array shape (N, 4), got {arr.shape}"
    )

    coords = arr[:, :3].astype(np.float32)
    occ = arr[:, 3].astype(np.float32)

    return coords, occ