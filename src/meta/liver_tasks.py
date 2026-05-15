# src/meta/liver_tasks.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.occupancy_dataset import load_case, normalize_case_id


@dataclass
class TaskBatchInfo:
    """
    Metadata describing which liver tasks were used in one outer-loop update.
    """
    task_epoch: int
    batch_start: int
    batch_end: int
    case_names: list[str]
    npy_paths: list[str]


class LiverTask:
    """
    One liver object = one MAML task.

    Inner loop:
        Adapt the initialization using support points from this liver.

    Query loss:
        Evaluate the adapted parameters using query points from this same liver.
    """

    def __init__(
        self,
        npy_path: Path,
        balanced: bool = True,
        seed: int | None = None,
    ):
        self.npy_path = Path(npy_path)
        self.case_name = normalize_case_id(self.npy_path.name)
        self.coords, self.occ = load_case(self.npy_path)
        self.balanced = balanced

        self.inside_idx = np.where(self.occ >= 0.5)[0]
        self.outside_idx = np.where(self.occ < 0.5)[0]

        self.rng = np.random.default_rng(seed)

    @property
    def num_points(self) -> int:
        return int(len(self.coords))

    @property
    def inside_ratio(self) -> float:
        return float(np.mean(self.occ >= 0.5))

    def sample_data(self, size, device: torch.device):
        """
        Sample support/query points from this liver.

        size can be:
            int     -> sample that many points
            "all"   -> use all available points exactly once
            None    -> use all available points exactly once
        """

        use_all = size is None or size == "all"

        if isinstance(size, str) and size != "all":
            raise ValueError(f"Unknown sample size string: {size}")

        if isinstance(size, int) and size >= len(self.coords):
            use_all = True

        if use_all:
            idx = np.arange(len(self.coords))
            self.rng.shuffle(idx)

        elif self.balanced and len(self.inside_idx) > 0 and len(self.outside_idx) > 0:
            n_inside = size // 2
            n_outside = size - n_inside

            replace_inside = n_inside > len(self.inside_idx)
            replace_outside = n_outside > len(self.outside_idx)

            idx_inside = self.rng.choice(
                self.inside_idx,
                n_inside,
                replace=replace_inside,
            )
            idx_outside = self.rng.choice(
                self.outside_idx,
                n_outside,
                replace=replace_outside,
            )

            idx = np.concatenate([idx_inside, idx_outside])
            self.rng.shuffle(idx)

        else:
            replace = size > len(self.coords)

            idx = self.rng.choice(
                len(self.coords),
                size,
                replace=replace,
            )

        x = torch.from_numpy(self.coords[idx]).float().to(device)
        y = torch.from_numpy(self.occ[idx]).float().unsqueeze(1).to(device)

        return x, y

class LiverTaskDistribution:
    """
    Provides liver tasks for MAML.

    Main mode:
        sequential = shuffle the full task list once, then consume it in chunks.
        This guarantees systematic dataset coverage.

    Optional old-style mode:
        random = sample liver tasks with replacement.
        This is kept only for debugging/comparison.
    """

    def __init__(
        self,
        npy_paths: list[Path],
        balanced: bool = True,
        seed: int = 2024,
        sampling_mode: str = "sequential",
        preload_tasks: bool = False,
    ):
        self.npy_paths = [Path(p) for p in npy_paths]
        self.balanced = balanced
        self.seed = seed
        self.sampling_mode = sampling_mode.lower()
        self.preload_tasks = preload_tasks

        if len(self.npy_paths) == 0:
            raise ValueError("No npy files found for meta-training.")

        if self.sampling_mode not in {"sequential", "random"}:
            raise ValueError(
                f"Unknown sampling_mode={sampling_mode}. "
                "Use 'sequential' or 'random'."
            )

        self.rng = np.random.default_rng(seed)
        self.task_cache: dict[Path, LiverTask] = {}

        self.task_epoch = 0
        self.cursor = 0
        self.order = np.arange(len(self.npy_paths))
        self._reshuffle_order()

        if self.preload_tasks:
            self.preload_all_tasks()

    def __len__(self) -> int:
        return len(self.npy_paths)

    def _reshuffle_order(self):
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.task_epoch += 1

    def _task_seed(self, npy_path: Path) -> int:
        """
        Stable-ish per-task seed generated from base seed and case name.
        """
        case_name = normalize_case_id(npy_path.name)
        return abs(hash((self.seed, case_name))) % (2**32)

    def get_task(self, npy_path: Path) -> LiverTask:
        """
        Load a task once and reuse it afterward.
        """
        npy_path = Path(npy_path)

        if npy_path not in self.task_cache:
            self.task_cache[npy_path] = LiverTask(
                npy_path=npy_path,
                balanced=self.balanced,
                seed=self._task_seed(npy_path),
            )

        return self.task_cache[npy_path]

    def preload_all_tasks(self):
        """
        Load all liver tasks into CPU memory.

        Faster training, but higher RAM usage.
        """
        for p in self.npy_paths:
            self.get_task(p)

    def next_task_batch(self, batch_size: int) -> tuple[list[LiverTask], TaskBatchInfo]:
        """
        Return a batch of liver tasks for one outer-loop meta-update.

        sequential mode:
            consumes shuffled task list without replacement.
            reshuffles only after all tasks have been used.

        random mode:
            old behavior; samples with replacement.
        """
        if batch_size <= 0:
            raise ValueError("batch_size / num_metatasks must be positive.")

        if self.sampling_mode == "random":
            chosen_indices = self.rng.choice(
                len(self.npy_paths),
                size=batch_size,
                replace=True,
            )

            chosen_paths = [self.npy_paths[int(i)] for i in chosen_indices]
            tasks = [self.get_task(p) for p in chosen_paths]

            info = TaskBatchInfo(
                task_epoch=self.task_epoch,
                batch_start=-1,
                batch_end=-1,
                case_names=[t.case_name for t in tasks],
                npy_paths=[str(t.npy_path) for t in tasks],
            )

            return tasks, info

        # sequential mode
        chosen_paths: list[Path] = []
        batch_start = self.cursor

        while len(chosen_paths) < batch_size:
            remaining = len(self.order) - self.cursor
            need = batch_size - len(chosen_paths)

            take = min(remaining, need)

            selected_indices = self.order[self.cursor : self.cursor + take]
            chosen_paths.extend([self.npy_paths[int(i)] for i in selected_indices])

            self.cursor += take

            if self.cursor >= len(self.order):
                self._reshuffle_order()

        batch_end = self.cursor
        tasks = [self.get_task(p) for p in chosen_paths]

        info = TaskBatchInfo(
            task_epoch=self.task_epoch,
            batch_start=batch_start,
            batch_end=batch_end,
            case_names=[t.case_name for t in tasks],
            npy_paths=[str(t.npy_path) for t in tasks],
        )

        return tasks, info

    def dataset_summary(self) -> dict:
        return {
            "num_tasks": len(self.npy_paths),
            "sampling_mode": self.sampling_mode,
            "balanced": self.balanced,
            "preload_tasks": self.preload_tasks,
            "seed": self.seed,
        }