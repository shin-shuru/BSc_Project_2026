import argparse
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.occupancy_dataset import (
    build_mesh_lookup,
    find_all_npy_files,
    load_case,
    match_npy_to_mesh,
    normalize_case_id,
    select_npy_files_by_split,
)
from src.evaluation.evaluator import evaluate_case
from src.models.factory import build_model
from src.reconstruction.marching_cubes import reconstruct_mesh
from src.training.trainer import train_single_case
from src.utils.io import save_case_outputs, to_serializable
from src.utils.metrics_patch import apply_fscore_patch
from src.utils.paths import (
    MESH_ROOT,
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
        return yaml.safe_load(f)


def run_one_case(
    npy_path: Path,
    gt_mesh_path: Path,
    output_root: Path,
    device: torch.device,
    config: dict,
):
    model_cfg = config["model"]
    training_cfg = config["training"]
    recon_cfg = config["reconstruction"]
    eval_cfg = config["evaluation"]

    seed = training_cfg.get("seed", 2024)

    case_name = normalize_case_id(npy_path.name)
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    coords, occ = load_case(npy_path)

    model = build_model(config)

    model, loss_history = train_single_case(
        model=model,
        coords=coords,
        occ=occ,
        device=device,
        seed=seed,
        case_name=case_name,
        batch_size=training_cfg.get("batch_size", 2048),
        lr=training_cfg.get("lr", 1e-3),
        num_epochs=training_cfg.get("epochs", 600),
    )

    pred_mesh = reconstruct_mesh(
        model=model,
        device=device,
        grid_res=recon_cfg.get("grid_res", 256),
        grid_min=recon_cfg.get("grid_min", -0.5),
        grid_max=recon_cfg.get("grid_max", 0.5),
        grid_batch_size=recon_cfg.get("grid_batch_size", 200_000),
    )

    metrics = evaluate_case(
        pred_mesh=pred_mesh,
        gt_mesh_path=gt_mesh_path,
        seed=seed,
        n_cube=eval_cfg.get("n_cube", 256),
        n_pointcloud=eval_cfg.get("n_pointcloud", 100_000),
        hash_resolution=eval_cfg.get("hash_resolution", 512),
    )

    metrics["case_name"] = case_name
    metrics["seed"] = seed
    metrics["npy_path"] = str(npy_path)
    metrics["gt_mesh_path"] = str(gt_mesh_path)
    metrics["final_loss"] = float(loss_history[-1])

    metrics["model_type"] = model_cfg["type"]
    metrics["hidden_features"] = model_cfg.get("hidden_features")
    metrics["hidden_layers"] = model_cfg.get("hidden_layers")

    if model_cfg["type"].lower() == "siren":
        metrics["first_w0"] = model_cfg.get("first_w0")
        metrics["hidden_w0"] = model_cfg.get("hidden_w0")

    metrics["lr"] = training_cfg.get("lr")
    metrics["epochs"] = training_cfg.get("epochs")
    metrics["batch_size"] = training_cfg.get("batch_size")
    metrics["grid_res"] = recon_cfg.get("grid_res")

    save_case_outputs(
        case_dir=case_dir,
        model=model,
        pred_mesh=pred_mesh,
        loss_history=loss_history,
        metrics=metrics,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Batch INR training and evaluation for ReLU/SIREN models."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file.",
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
        help="Only run first N selected cases.",
    )

    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=None,
        help="Optional explicit case IDs to run.",
    )

    args = parser.parse_args()

    config = load_config(Path(args.config))

    apply_fscore_patch()

    training_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    seed = training_cfg.get("seed", 2024)
    seed_all(seed)

    device = get_device(args.device)
    print("Using device:", device)

    npy_root = Path(data_cfg.get("npy_root", NPY_ROOT))
    mesh_root = Path(data_cfg.get("mesh_root", MESH_ROOT))

    train_split = Path(data_cfg.get("train_split", TRAIN_SPLIT))
    val_split = Path(data_cfg.get("val_split", VAL_SPLIT))
    test_split = Path(data_cfg.get("test_split", TEST_SPLIT))

    split = data_cfg.get("split", "test")

    run_name = output_cfg.get("run_name", "debug_run")
    output_root = Path(output_cfg.get("output_root", OUTPUT_ROOT / "runs" / run_name))
    output_root.mkdir(parents=True, exist_ok=True)

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

    mesh_lookup = build_mesh_lookup(mesh_root)

    print(f"Found {len(all_npy_files)} total npy files")
    print(f"Running {len(selected_npy_files)} npy files from split='{split}'")
    print(f"Output root: {output_root}")

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

        print(f"\n[{i + 1}/{total_cases}] {npy_path.name}")

        try:
            metrics = run_one_case(
                npy_path=npy_path,
                gt_mesh_path=gt_mesh_path,
                output_root=output_root,
                device=device,
                config=config,
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
                "seed": seed,
            })

        pd.DataFrame(summary_rows).to_csv(output_root / "summary.csv", index=False)

    print("\nDone.")
    print("Summary saved to:", output_root / "summary.csv")


if __name__ == "__main__":
    main()