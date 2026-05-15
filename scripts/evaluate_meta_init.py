# scripts/evaluate_meta_init.py

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.occupancy_dataset import (
    OccupancyDataset,
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
from src.utils.io import to_serializable
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


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty or invalid: {path}")

    return config


def save_yaml(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_serializable(config), f, sort_keys=False)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_meta_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    return model


def get_eval_epochs_for_init(config: dict, init_type: str) -> list[int]:
    training_cfg = config["training"]
    eval_cfg = config["evaluation"]

    max_epochs = int(training_cfg.get("epochs", 600))

    if init_type == "meta":
        raw_epochs = eval_cfg.get("meta_eval_epochs", eval_cfg.get("eval_epochs", [0, max_epochs]))
    elif init_type == "random":
        raw_epochs = eval_cfg.get("random_eval_epochs", [max_epochs])
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    eval_epochs = sorted(set(int(e) for e in raw_epochs if 0 <= int(e) <= max_epochs))

    if init_type == "random" and len(eval_epochs) == 0:
        eval_epochs = [max_epochs]

    if init_type == "meta" and len(eval_epochs) == 0:
        eval_epochs = [0, max_epochs]

    return eval_epochs


def save_snapshot_outputs(
    snapshot_dir: Path,
    model: nn.Module,
    pred_mesh,
    metrics: dict,
):
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), snapshot_dir / "model.pt")
    pred_mesh.export(snapshot_dir / "pred_mesh.stl")
    save_json(snapshot_dir / "metrics.json", metrics)


def evaluate_snapshot(
    model: nn.Module,
    device: torch.device,
    gt_mesh_path: Path,
    config: dict,
    seed: int,
):
    recon_cfg = config["reconstruction"]
    eval_cfg = config["evaluation"]

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

    return pred_mesh, metrics


def add_common_metric_fields(
    metrics: dict,
    *,
    config: dict,
    npy_path: Path,
    gt_mesh_path: Path,
    case_name: str,
    init_type: str,
    eval_epoch: int,
    max_epochs: int,
    seed: int,
    lr: float,
    batch_size: int,
    epoch_loss: float | None,
    init_checkpoint: Path | None,
):
    model_cfg = config["model"]
    recon_cfg = config["reconstruction"]

    metrics["case_name"] = case_name
    metrics["init_type"] = init_type
    metrics["eval_epoch"] = int(eval_epoch)
    metrics["epochs"] = int(eval_epoch)
    metrics["max_epochs"] = int(max_epochs)

    metrics["seed"] = int(seed)
    metrics["npy_path"] = str(npy_path)
    metrics["gt_mesh_path"] = str(gt_mesh_path)

    metrics["model_type"] = model_cfg["type"]
    metrics["hidden_features"] = model_cfg.get("hidden_features")
    metrics["hidden_layers"] = model_cfg.get("hidden_layers")

    if model_cfg["type"].lower() == "siren":
        metrics["first_w0"] = model_cfg.get("first_w0")
        metrics["hidden_w0"] = model_cfg.get("hidden_w0")

    metrics["lr"] = float(lr)
    metrics["batch_size"] = int(batch_size)
    metrics["grid_res"] = recon_cfg.get("grid_res")
    metrics["final_loss"] = float(epoch_loss) if epoch_loss is not None else None
    metrics["init_checkpoint"] = str(init_checkpoint) if init_checkpoint is not None else None
    metrics["status"] = "ok"

    return metrics


def run_one_case_one_init(
    *,
    npy_path: Path,
    gt_mesh_path: Path,
    init_type: str,
    output_root: Path,
    device: torch.device,
    config: dict,
):
    training_cfg = config["training"]

    seed = int(training_cfg.get("seed", 2024))
    batch_size = int(training_cfg.get("batch_size", 2048))
    lr = float(training_cfg.get("lr", 1e-3))
    max_epochs = int(training_cfg.get("epochs", 600))
    num_workers = int(training_cfg.get("num_workers", 0))

    eval_epochs = get_eval_epochs_for_init(config, init_type)
    eval_epoch_set = set(eval_epochs)

    case_name = normalize_case_id(npy_path.name)
    case_output_root = output_root / init_type / case_name
    case_output_root.mkdir(parents=True, exist_ok=True)

    # Reset seed per case/init for reproducibility.
    torch_gen, _ = seed_all(seed)

    coords, occ = load_case(npy_path)

    dataset = OccupancyDataset(coords, occ)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch_gen,
    )

    model = build_model(config).to(device)

    init_checkpoint = None

    if init_type == "meta":
        init_checkpoint = resolve_repo_path(config["meta_checkpoint"]["path"])
        model = load_meta_checkpoint(model, init_checkpoint, device)

    elif init_type == "random":
        init_checkpoint = None

    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []
    rows: list[dict] = []

    def record_snapshot(epoch: int, epoch_loss: float | None):
        print(f"    Evaluating {init_type} | {case_name} | epoch {epoch}")

        pred_mesh, metrics = evaluate_snapshot(
            model=model,
            device=device,
            gt_mesh_path=gt_mesh_path,
            config=config,
            seed=seed,
        )

        metrics = add_common_metric_fields(
            metrics,
            config=config,
            npy_path=npy_path,
            gt_mesh_path=gt_mesh_path,
            case_name=case_name,
            init_type=init_type,
            eval_epoch=epoch,
            max_epochs=max_epochs,
            seed=seed,
            lr=lr,
            batch_size=batch_size,
            epoch_loss=epoch_loss,
            init_checkpoint=init_checkpoint,
        )

        snapshot_dir = case_output_root / f"epoch_{epoch:04d}"

        save_snapshot_outputs(
            snapshot_dir=snapshot_dir,
            model=model,
            pred_mesh=pred_mesh,
            metrics=metrics,
        )

        rows.append(to_serializable(metrics))

    # Epoch 0 = evaluate raw initialization before any adaptation.
    # This is normally useful for meta init, but supported for random if requested.
    if 0 in eval_epoch_set:
        record_snapshot(epoch=0, epoch_loss=None)

    train_eval_epochs = set(e for e in eval_epochs if e > 0)

    epoch_bar = tqdm(
        range(1, max_epochs + 1),
        desc=f"{init_type}:{case_name}",
        unit="epoch",
        leave=False,
    )

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
        loss_history.append(float(epoch_loss))

        epoch_bar.set_postfix(loss=f"{epoch_loss:.6f}")

        if epoch in train_eval_epochs:
            record_snapshot(epoch=epoch, epoch_loss=epoch_loss)

    np.save(
        case_output_root / "loss_history.npy",
        np.array(loss_history, dtype=np.float32),
    )

    return rows


def summarize_by_init_and_epoch(summary_csv: Path):
    df = pd.read_csv(summary_csv)
    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) == 0:
        print("No successful rows found for summary.")
        return

    metric_cols = [
        "OccNet chamfer-L1",
        "VIoU",
        "OccNet f-scores",
        "final_loss",
    ]

    rows = []

    grouped = df_ok.groupby(["init_type", "eval_epoch"], dropna=False)

    for (init_type, eval_epoch), group in grouped:
        row = {
            "init_type": init_type,
            "eval_epoch": int(eval_epoch),
            "n": len(group),
        }

        for col in metric_cols:
            if col in group.columns:
                row[f"mean_{col}"] = group[col].mean()
                row[f"std_{col}"] = group[col].std()

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["eval_epoch", "init_type"])

    out_path = summary_csv.parent / "comparison_summary.csv"
    out_df.to_csv(out_path, index=False)

    print("\nComparison summary:")
    print(out_df)
    print("\nSaved to:", out_path)


def summarize_meta_against_random_final(summary_csv: Path):
    """
    Create an additional table comparing each meta evaluation epoch against
    the final random baseline for the same case.

    This is useful when the random baseline is evaluated only once at the end.
    """
    df = pd.read_csv(summary_csv)
    df_ok = df[df["status"] == "ok"].copy()

    if len(df_ok) == 0:
        return

    meta_df = df_ok[df_ok["init_type"] == "meta"].copy()
    random_df = df_ok[df_ok["init_type"] == "random"].copy()

    if len(meta_df) == 0 or len(random_df) == 0:
        return

    # If random has only one row per case, this is simply that row.
    # If multiple random rows exist, use the largest eval_epoch as the final baseline.
    random_final = (
        random_df.sort_values("eval_epoch")
        .groupby("case_name", as_index=False)
        .tail(1)
        .copy()
    )

    metric_cols = [
        "OccNet chamfer-L1",
        "VIoU",
        "OccNet f-scores",
        "final_loss",
    ]

    keep_random_cols = ["case_name", "eval_epoch"] + [
        col for col in metric_cols if col in random_final.columns
    ]

    random_final = random_final[keep_random_cols].rename(
        columns={
            "eval_epoch": "random_eval_epoch",
            "OccNet chamfer-L1": "random_final_chamfer_l1",
            "VIoU": "random_final_viou",
            "OccNet f-scores": "random_final_fscore",
            "final_loss": "random_final_loss",
        }
    )

    merged = meta_df.merge(random_final, on="case_name", how="inner")

    rows = []

    for eval_epoch, group in merged.groupby("eval_epoch"):
        row = {
            "meta_eval_epoch": int(eval_epoch),
            "random_eval_epoch": int(group["random_eval_epoch"].iloc[0]),
            "n": len(group),
        }

        if "OccNet chamfer-L1" in group.columns and "random_final_chamfer_l1" in group.columns:
            row["mean_meta_chamfer_l1"] = group["OccNet chamfer-L1"].mean()
            row["mean_random_final_chamfer_l1"] = group["random_final_chamfer_l1"].mean()
            row["mean_chamfer_l1_delta_meta_minus_random"] = (
                group["OccNet chamfer-L1"] - group["random_final_chamfer_l1"]
            ).mean()

        if "VIoU" in group.columns and "random_final_viou" in group.columns:
            row["mean_meta_viou"] = group["VIoU"].mean()
            row["mean_random_final_viou"] = group["random_final_viou"].mean()
            row["mean_viou_delta_meta_minus_random"] = (
                group["VIoU"] - group["random_final_viou"]
            ).mean()

        if "OccNet f-scores" in group.columns and "random_final_fscore" in group.columns:
            row["mean_meta_fscore"] = group["OccNet f-scores"].mean()
            row["mean_random_final_fscore"] = group["random_final_fscore"].mean()
            row["mean_fscore_delta_meta_minus_random"] = (
                group["OccNet f-scores"] - group["random_final_fscore"]
            ).mean()

        if "final_loss" in group.columns and "random_final_loss" in group.columns:
            row["mean_meta_loss"] = group["final_loss"].mean()
            row["mean_random_final_loss"] = group["random_final_loss"].mean()
            row["mean_loss_delta_meta_minus_random"] = (
                group["final_loss"] - group["random_final_loss"]
            ).mean()

        rows.append(row)

    out_df = pd.DataFrame(rows)

    if len(out_df) == 0:
        return

    out_df = out_df.sort_values("meta_eval_epoch")

    out_path = summary_csv.parent / "meta_vs_random_final_summary.csv"
    out_df.to_csv(out_path, index=False)

    print("\nMeta vs random-final summary:")
    print(out_df)
    print("\nSaved to:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate meta-learned SIREN initialization with intermediate snapshots, "
            "and compare against final random SIREN initialization."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_meta_init_siren.yaml",
        help="Evaluation config YAML.",
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
        help="Optional number of test cases for debugging.",
    )

    parser.add_argument(
        "--case_ids",
        nargs="*",
        default=None,
        help="Optional explicit case IDs to evaluate.",
    )

    parser.add_argument(
        "--init_types",
        nargs="*",
        default=["random", "meta"],
        choices=["random", "meta"],
        help="Which initialization types to evaluate.",
    )

    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    config = load_yaml(config_path)

    apply_fscore_patch()

    training_cfg = config["training"]
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})

    seed = int(training_cfg.get("seed", 2024))
    seed_all(seed)

    device = get_device(args.device)

    run_name = output_cfg.get("run_name", "siren_meta_vs_random_single_run")
    output_root = Path(
        output_cfg.get(
            "output_root",
            OUTPUT_ROOT / "meta_vs_random_comparison" / run_name,
        )
    )

    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root

    output_root.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility.
    save_yaml(config, output_root / "config_used.yaml")
    shutil.copy2(config_path, output_root / "original_config.yaml")

    npy_root = Path(data_cfg.get("npy_root", NPY_ROOT))
    mesh_root = Path(data_cfg.get("mesh_root", MESH_ROOT))

    train_split = Path(data_cfg.get("train_split", TRAIN_SPLIT))
    val_split = Path(data_cfg.get("val_split", VAL_SPLIT))
    test_split = Path(data_cfg.get("test_split", TEST_SPLIT))

    split = data_cfg.get("split", "test")

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

    print("Using device:", device)
    print("Split:", split)
    print("Total npy files:", len(all_npy_files))
    print("Selected cases:", len(selected_npy_files))
    print("Init types:", args.init_types)
    print("Meta eval epochs:", get_eval_epochs_for_init(config, "meta"))
    print("Random eval epochs:", get_eval_epochs_for_init(config, "random"))
    print("Output root:", output_root)

    all_rows: list[dict] = []
    summary_csv = output_root / "summary.csv"

    total_cases = len(selected_npy_files)

    for i, npy_path in enumerate(tqdm(selected_npy_files, desc="Cases", unit="case")):
        case_name = normalize_case_id(npy_path.name)
        gt_mesh_path = match_npy_to_mesh(npy_path, mesh_lookup)

        if gt_mesh_path is None:
            print(f"[SKIP] No matching mesh for {npy_path.name}")

            all_rows.append({
                "case_name": case_name,
                "status": "missing_gt_mesh",
                "npy_path": str(npy_path),
            })

            pd.DataFrame(all_rows).to_csv(summary_csv, index=False)
            continue

        print(f"\n[{i + 1}/{total_cases}] {case_name}")

        for init_type in args.init_types:
            try:
                rows = run_one_case_one_init(
                    npy_path=npy_path,
                    gt_mesh_path=gt_mesh_path,
                    init_type=init_type,
                    output_root=output_root,
                    device=device,
                    config=config,
                )

                all_rows.extend(rows)

            except Exception as e:
                print(f"[ERROR] {case_name} | {init_type}: {e}")

                all_rows.append({
                    "case_name": case_name,
                    "init_type": init_type,
                    "status": "error",
                    "error": str(e),
                    "npy_path": str(npy_path),
                    "gt_mesh_path": str(gt_mesh_path),
                    "seed": seed,
                })

            pd.DataFrame(all_rows).to_csv(summary_csv, index=False)

    print("\nDone.")
    print("Summary saved to:", summary_csv)

    summarize_by_init_and_epoch(summary_csv)
    summarize_meta_against_random_final(summary_csv)


if __name__ == "__main__":
    main()