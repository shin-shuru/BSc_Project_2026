import argparse
from pathlib import Path
import sys

import torch
import yaml
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.occupancy_dataset import (
    find_all_npy_files,
    select_npy_files_by_split,
)
from src.meta.liver_tasks import LiverTaskDistribution
from src.meta.maml import LiverMAML
from src.models.factory import build_model
from src.utils.paths import (
    NPY_ROOT,
    OUTPUT_ROOT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
)
from src.utils.seed import seed_all


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Meta-train INR initialization using MAML."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to meta config YAML.",
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

    args = parser.parse_args()

    config = load_config(Path(args.config))
    meta_cfg = config["meta"]
    output_cfg = config["output"]

    seed = meta_cfg.get("seed", 2024)
    seed_all(seed)

    device = get_device(args.device)
    print("Using device:", device)

    npy_root = Path(meta_cfg.get("npy_root", NPY_ROOT))
    split = meta_cfg.get("split", "train")

    all_npy_files = find_all_npy_files(npy_root)

    selected_npy_files = select_npy_files_by_split(
        npy_files=all_npy_files,
        split=split,
        train_split=Path(meta_cfg.get("train_split", TRAIN_SPLIT)),
        val_split=Path(meta_cfg.get("val_split", VAL_SPLIT)),
        test_split=Path(meta_cfg.get("test_split", TEST_SPLIT)),
        case_ids=None,
    )

    if args.limit is not None:
        selected_npy_files = selected_npy_files[: args.limit]

    print(f"Found {len(all_npy_files)} total npy files")
    print(f"Using {len(selected_npy_files)} files for meta split='{split}'")

    task_distribution = LiverTaskDistribution(
        selected_npy_files,
        balanced=meta_cfg.get("balanced_sampling", True),
    )

    model = build_model(config)

    maml = LiverMAML(
        model=model,
        task_distribution=task_distribution,
        device=device,
        alpha=meta_cfg.get("alpha", 5e-4),
        beta=meta_cfg.get("beta", 5e-5),
        k_support=meta_cfg.get("k_support", 4096),
        k_query=meta_cfg.get("k_query", 4096),
        num_metatasks=meta_cfg.get("num_metatasks", 4),
        inner_steps=meta_cfg.get("inner_steps", 1),
        first_order=meta_cfg.get("first_order", False),
    )

    meta_losses = maml.outer_loop(
        num_iterations=meta_cfg.get("iterations", 10000),
        print_every=meta_cfg.get("print_every", 50),
    )

    run_name = output_cfg.get("run_name", "meta_siren")
    output_root = Path(output_cfg.get("output_root", OUTPUT_ROOT / "meta" / run_name))
    output_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_root / "meta_init.pt"
    loss_path = output_root / "meta_losses.npy"

    torch.save(
        {
            "model_state_dict": maml.model.state_dict(),
            "config": config,
            "meta_losses": meta_losses,
        },
        ckpt_path,
    )

    np.save(loss_path, np.array(meta_losses, dtype=np.float32))

    print("Saved meta initialization to:", ckpt_path)
    print("Saved meta losses to:", loss_path)


if __name__ == "__main__":
    main()