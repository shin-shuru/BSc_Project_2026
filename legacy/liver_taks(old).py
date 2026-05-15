from pathlib import Path

import numpy as np
import torch

from src.data.occupancy_dataset import load_case


class LiverTask:
    """
    One liver object = one MAML task.
    The task samples support/query occupancy points from one .npy case.
    """

    def __init__(self, npy_path: Path, balanced: bool = True):
        self.npy_path = Path(npy_path)
        self.coords, self.occ = load_case(self.npy_path)
        self.balanced = balanced

        self.inside_idx = np.where(self.occ >= 0.5)[0]
        self.outside_idx = np.where(self.occ < 0.5)[0]

    def sample_data(self, size: int, device: torch.device):
        if self.balanced and len(self.inside_idx) > 0 and len(self.outside_idx) > 0:
            n_inside = size // 2
            n_outside = size - n_inside

            idx_inside = np.random.choice(self.inside_idx, n_inside, replace=True)
            idx_outside = np.random.choice(self.outside_idx, n_outside, replace=True)
            idx = np.concatenate([idx_inside, idx_outside])
            np.random.shuffle(idx)
        else:
            idx = np.random.choice(len(self.coords), size, replace=True)

        x = torch.from_numpy(self.coords[idx]).float().to(device)
        y = torch.from_numpy(self.occ[idx]).float().unsqueeze(1).to(device)

        return x, y


class LiverTaskDistribution:
    """
    Equivalent to SineDistribution in the supervisor notebook.
    Randomly samples liver tasks from the train split.
    """

    def __init__(self, npy_paths: list[Path], balanced: bool = True):
        self.npy_paths = [Path(p) for p in npy_paths]
        self.balanced = balanced

        if len(self.npy_paths) == 0:
            raise ValueError("No npy files found for meta-training.")

    def sample_task(self) -> LiverTask:
        npy_path = np.random.choice(self.npy_paths)
        return LiverTask(npy_path, balanced=self.balanced)