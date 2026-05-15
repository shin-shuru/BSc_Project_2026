from __future__ import annotations

import csv
import json
import platform
import resource
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from src.utils.io import to_serializable


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_cpu_memory_mb() -> float:
    """
    Returns max resident set size in MB on Linux.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Linux returns KB; macOS returns bytes.
    if platform.system().lower() == "darwin":
        return usage / (1024 ** 2)

    return usage / 1024


def get_gpu_memory_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_reserved_mb": None,
            "gpu_memory_max_allocated_mb": None,
            "gpu_memory_max_reserved_mb": None,
        }

    return {
        "gpu_memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "gpu_memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "gpu_memory_max_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "gpu_memory_max_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024 ** 2),
    }


class MetaLogger:
    """
    Writes meta-training logs continuously during training.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_log_path = self.output_dir / "meta_training_log.csv"
        self.task_batches_path = self.output_dir / "task_batches.csv"

        self._training_header_written = False
        self._task_header_written = False

    def append_training_row(self, row: dict[str, Any]):
        row = to_serializable(row)

        write_header = not self.training_log_path.exists() or not self._training_header_written

        with open(self.training_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            if write_header:
                writer.writeheader()
                self._training_header_written = True

            writer.writerow(row)

    def append_task_batch(self, iteration: int, task_batch_info):
        rows = []

        for case_name, npy_path in zip(
            task_batch_info.case_names,
            task_batch_info.npy_paths,
        ):
            rows.append({
                "iteration": iteration,
                "task_epoch": task_batch_info.task_epoch,
                "batch_start": task_batch_info.batch_start,
                "batch_end": task_batch_info.batch_end,
                "case_name": case_name,
                "npy_path": npy_path,
            })

        if not rows:
            return

        write_header = not self.task_batches_path.exists() or not self._task_header_written

        with open(self.task_batches_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))

            if write_header:
                writer.writeheader()
                self._task_header_written = True

            writer.writerows(rows)


class LiverMAML:
    """
    MAML trainer for occupancy INR.

    Terminology:
        initialization:
            The model parameters before task-specific adaptation.

        inner loop:
            For one liver task, adapt the initialization using support points.

        outer loop:
            Use query losses from multiple liver tasks to update the shared
            initialization.

        num_metatasks:
            Number of liver tasks contributing to one outer-loop meta-update.
    """

    def __init__(
        self,
        model: nn.Module,
        task_distribution,
        device: torch.device,
        alpha: float = 1e-3,
        beta: float = 1e-4,
        k_support: int | str | None = 4096,
        k_query: int | str | None = 4096,
        num_metatasks: int = 4,
        inner_steps: int = 1,
        first_order: bool = False,
        output_dir: Path | None = None,
        checkpoint_every: int = 0,
        model_config: dict | None = None,
    ):
        self.model = model.to(device)
        self.task_distribution = task_distribution
        self.device = device

        self.alpha = alpha
        self.beta = beta
        self.k_support = k_support
        self.k_query = k_query
        self.num_metatasks = num_metatasks
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.checkpoint_every = checkpoint_every
        self.model_config = model_config or {}

        self.criterion = nn.BCEWithLogitsLoss()
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=beta)
        self.meta_losses: list[float] = []

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.logger = MetaLogger(self.output_dir) if self.output_dir is not None else None

        if self.output_dir is not None:
            (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def named_parameters_dict(self):
        return OrderedDict(
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        )

    def forward_with_params(self, x, params):
        return functional_call(self.model, params, (x,))

    def inner_loop(self, task):
        """
        Inner loop for one liver task.

        1. Start from current initialization.
        2. Adapt using support points.
        3. Evaluate adapted parameters on query points.
        """
        params = self.named_parameters_dict()
        
        # The `task` here is liver
        
        for _ in range(self.inner_steps):
            # Sample some points: 4096 eg
            x_support, y_support = task.sample_data(self.k_support, self.device)

            support_logits = self.forward_with_params(x_support, params)
            support_loss = self.criterion(support_logits, y_support)

            grads = torch.autograd.grad(
                support_loss,
                params.values(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order,
            )

            # Temp update the weights
            params = OrderedDict(
                (name, param - self.alpha * grad)
                for (name, param), grad in zip(params.items(), grads)
            )

        # Cal the loss based on the temp_updated_weight for the meta loss
        x_query, y_query = task.sample_data(self.k_query, self.device)
        query_logits = self.forward_with_params(x_query, params)
        query_loss = self.criterion(query_logits, y_query)

        return query_loss

    def save_checkpoint(self, iteration: int, final: bool = False):
        if self.output_dir is None:
            return

        ckpt_name = "meta_model_final.pt" if final else f"meta_iter_{iteration:06d}.pt"
        ckpt_path = self.output_dir / "checkpoints" / ckpt_name

        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.meta_optimizer.state_dict(),
                "meta_losses": self.meta_losses,
                "alpha": self.alpha,
                "beta": self.beta,
                "k_support": self.k_support,
                "k_query": self.k_query,
                "num_metatasks": self.num_metatasks,
                "inner_steps": self.inner_steps,
                "first_order": self.first_order,
                "model_config": self.model_config,
            },
            ckpt_path,
        )

    def save_loss_history(self):
        if self.output_dir is None:
            return

        np.save(
            self.output_dir / "meta_loss_history.npy",
            np.array(self.meta_losses, dtype=np.float32),
        )

        with open(self.output_dir / "meta_loss_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["iteration", "meta_loss"])
            writer.writeheader()

            for i, loss in enumerate(self.meta_losses, start=1):
                writer.writerow({
                    "iteration": i,
                    "meta_loss": loss,
                })

    def save_run_info(self, run_info: dict):
        if self.output_dir is None:
            return

        with open(self.output_dir / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(to_serializable(run_info), f, indent=2)

    def outer_loop(
        self,
        num_iterations: int,
        print_every: int = 10,
        log_every: int = 1,
    ):
        self.model.train()

        total_start_time = time.perf_counter()

        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for iteration in range(1, num_iterations + 1):
            iter_start_time = time.perf_counter()

            self.meta_optimizer.zero_grad(set_to_none=True)

            # Take some livers based on batch size
            tasks, task_batch_info = self.task_distribution.next_task_batch(
                self.num_metatasks
            )

            task_losses = []
            
            # Iterate each liver
            for task in tasks:
                task_loss = self.inner_loop(task)
                task_losses.append(task_loss)

            meta_loss = torch.stack(task_losses).mean()
            meta_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.meta_optimizer.step()

            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize(self.device)

            loss_value = float(meta_loss.detach().cpu())
            self.meta_losses.append(loss_value)

            iter_end_time = time.perf_counter()
            elapsed_time_sec = iter_end_time - total_start_time
            iteration_time_sec = iter_end_time - iter_start_time

            if self.logger is not None and iteration % log_every == 0:
                gpu_stats = get_gpu_memory_stats(self.device)

                log_row = {
                    "iteration": iteration,
                    "task_epoch": task_batch_info.task_epoch,
                    "task_batch_start": task_batch_info.batch_start,
                    "task_batch_end": task_batch_info.batch_end,
                    "num_tasks_used": len(tasks),
                    "meta_loss": loss_value,
                    "elapsed_time_sec": elapsed_time_sec,
                    "iteration_time_sec": iteration_time_sec,
                    "cpu_memory_rss_mb": get_cpu_memory_mb(),
                    **gpu_stats,
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "k_support": self.k_support,
                    "k_query": self.k_query,
                    "inner_steps": self.inner_steps,
                    "first_order": self.first_order,
                    "model_param_count": count_trainable_parameters(self.model),
                    "model_type": self.model_config.get("type"),
                    "hidden_features": self.model_config.get("hidden_features"),
                    "hidden_layers": self.model_config.get("hidden_layers"),
                    "first_w0": self.model_config.get("first_w0"),
                    "hidden_w0": self.model_config.get("hidden_w0"),
                    "num_frequencies": self.model_config.get("num_frequencies"),
                    "include_input": self.model_config.get("include_input"),
                    "log_sampling": self.model_config.get("log_sampling"),
                    "pe_scale": self.model_config.get("pe_scale"),
                    "softplus_beta": self.model_config.get("softplus_beta"),
                    "softplus_threshold": self.model_config.get("softplus_threshold"),
                }

                self.logger.append_training_row(log_row)
                self.logger.append_task_batch(iteration, task_batch_info)

            if iteration % print_every == 0:
                print(
                    f"[{iteration}/{num_iterations}] "
                    f"meta_loss={loss_value:.6f} "
                    f"tasks={len(tasks)} "
                    f"task_epoch={task_batch_info.task_epoch} "
                    f"time={iteration_time_sec:.2f}s"
                )

            if self.checkpoint_every > 0 and iteration % self.checkpoint_every == 0:
                self.save_checkpoint(iteration=iteration, final=False)
                self.save_loss_history()

        self.save_checkpoint(iteration=num_iterations, final=True)
        self.save_loss_history()

        return self.meta_losses
