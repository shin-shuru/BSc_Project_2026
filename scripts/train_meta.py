# scripts/train_meta.py

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.occupancy_dataset import (
    find_all_npy_files,
    normalize_case_id,
    select_npy_files_by_split,
)
from src.meta.liver_tasks import LiverTaskDistribution
from src.meta.maml import LiverMAML, count_trainable_parameters
from src.models.factory import build_model
from src.utils.io import to_serializable
from src.utils.paths import (
    NPY_ROOT,
    OUTPUT_ROOT,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.utils.seed import seed_all


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return config


def save_config_snapshot(config: dict, config_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(config), f, sort_keys=False)

    shutil.copy2(config_path, output_dir / "original_config.yaml")


def save_selected_tasks(selected_npy_files: list[Path], output_dir: Path):
    out_path = output_dir / "selected_meta_tasks.csv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("index,case_name,npy_path\n")

        for i, p in enumerate(selected_npy_files):
            f.write(f"{i},{normalize_case_id(p.name)},{p}\n")


def get_gpu_name(device: torch.device) -> str | None:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Meta-train an INR initialization using MAML on liver occupancy tasks."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to meta-learning config YAML.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cuda, cuda:0, cpu.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use first N selected cases for debugging.",
    )

    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=None,
        help="Optional explicit case IDs to use for meta-training.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    model_cfg = config["model"]
    meta_cfg = config["meta"]
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    seed = int(meta_cfg.get("seed", 2024))
    seed_all(seed)

    device = get_device(args.device)

    npy_root = Path(data_cfg.get("npy_root", NPY_ROOT))
    train_split = Path(data_cfg.get("train_split", TRAIN_SPLIT))
    val_split = Path(data_cfg.get("val_split", VAL_SPLIT))
    test_split = Path(data_cfg.get("test_split", TEST_SPLIT))
    split = data_cfg.get("split", meta_cfg.get("split", "train"))

    run_name = output_cfg.get("run_name", "meta_maml_run")
    output_dir = Path(
        output_cfg.get(
            "output_dir",
            OUTPUT_ROOT / "meta_runs" / run_name,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(config, config_path, output_dir)

    all_npy_files = find_all_npy_files(npy_root)

    selected_npy_files = select_npy_files_by_split(
        npy_files=all_npy_files,
        split=split,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        case_ids=args.case_ids,
    )

    if args.limit is not None:
        selected_npy_files = selected_npy_files[: args.limit]

    if len(selected_npy_files) == 0:
        raise ValueError(
            f"No npy files selected. Check split={split}, npy_root={npy_root}, "
            f"and split files."
        )

    save_selected_tasks(selected_npy_files, output_dir)

    model = build_model(config)
    model_param_count = count_trainable_parameters(model)

    task_distribution = LiverTaskDistribution(
        npy_paths=selected_npy_files,
        balanced=bool(meta_cfg.get("balanced", True)),
        seed=seed,
        sampling_mode=meta_cfg.get("task_sampling", "sequential"),
        preload_tasks=bool(meta_cfg.get("preload_tasks", False)),
    )

    maml = LiverMAML(
        model=model,
        task_distribution=task_distribution,
        device=device,
        alpha=float(meta_cfg.get("alpha", 1e-3)),
        beta=float(meta_cfg.get("beta", 1e-4)),
        k_support=meta_cfg.get("k_support", 4096),
        k_query=meta_cfg.get("k_query", 4096),
        num_metatasks=int(meta_cfg.get("num_metatasks", 16)),
        inner_steps=int(meta_cfg.get("inner_steps", 1)),
        first_order=bool(meta_cfg.get("first_order", False)),
        output_dir=output_dir,
        checkpoint_every=int(meta_cfg.get("checkpoint_every", 0)),
        model_config=model_cfg,
    )

    run_info = {
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": get_gpu_name(device),
        "torch_version": torch.__version__,
        "n_total_npy_files": len(all_npy_files),
        "n_selected_meta_tasks": len(selected_npy_files),
        "split": split,
        "model_param_count": model_param_count,
        "task_distribution": task_distribution.dataset_summary(),
        "model_config": model_cfg,
        "meta_config": meta_cfg,
    }

    maml.save_run_info(run_info)

    print("Using device:", device)
    print("GPU:", get_gpu_name(device))
    print("Selected meta-training tasks:", len(selected_npy_files))
    print("Task sampling:", meta_cfg.get("task_sampling", "sequential"))
    print("num_metatasks:", meta_cfg.get("num_metatasks", 16))
    print("Model parameters:", model_param_count)
    print("Output dir:", output_dir)

    meta_losses = maml.outer_loop(
        num_iterations=int(meta_cfg.get("num_iterations", 1000)),
        print_every=int(meta_cfg.get("print_every", 10)),
        log_every=int(meta_cfg.get("log_every", 1)),
    )

    final_info = {
        **run_info,
        "end_time": datetime.now().isoformat(timespec="seconds"),
        "final_meta_loss": float(meta_losses[-1]) if meta_losses else None,
        "num_logged_losses": len(meta_losses),
    }

    with open(output_dir / "run_info_final.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(final_info), f, indent=2)

    print("\nDone.")
    print("Final meta loss:", final_info["final_meta_loss"])
    print("Saved outputs to:", output_dir)
    
    
    ### Pick one liver in valid set
    ### Run adaptation on one liver --> INR on one liver --> results
    ### inr_trainer = ....(model="siren", init_ckpt="meta_ckpt_path"|None means defaults init, save_path=meta_exp_path="adapt/s0004/epoch[1,2,5,10,50,100]", compare_with_random_path="Path_to_random_valid_results" )

if __name__ == "__main__":
    main()