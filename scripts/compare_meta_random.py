import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_base_config_from_meta_config(meta_config: dict, args) -> dict:
    if "model" not in meta_config:
        raise ValueError("Meta config must contain a model section.")

    return {
        "model": copy.deepcopy(meta_config["model"]),
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "checkpoint_epochs": args.checkpoint_epochs,
        },
        "reconstruction": {
            "grid_res": args.grid_res,
            "grid_min": args.grid_min,
            "grid_max": args.grid_max,
            "grid_batch_size": args.grid_batch_size,
        },
        "evaluation": {
            "n_cube": args.n_cube,
            "n_pointcloud": args.n_pointcloud,
            "hash_resolution": args.hash_resolution,
        },
        "data": {
            "split": args.split,
        },
        "output": {
            "run_name": args.comparison_name,
        },
    }


def run_command(cmd: list[str]):
    print("\nRunning:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def summarize_run(summary_path: Path, init_type: str) -> dict | None:
    if not summary_path.exists():
        return None

    df = pd.read_csv(summary_path)
    df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()

    if len(df_ok) == 0:
        return {
            "init_type": init_type,
            "n": 0,
            "status": "no_ok_cases",
        }

    row = {
        "init_type": init_type,
        "n": len(df_ok),
        "status": "ok",
    }

    for col in [
        "OccNet chamfer-L1",
        "OccNet chamfer-L2",
        "OccNet normal consistency",
        "OccNet f-scores",
        "VIoU",
        "Dice",
        "final_loss",
    ]:
        if col in df_ok.columns:
            row[f"mean_{col}"] = df_ok[col].mean()
            row[f"std_{col}"] = df_ok[col].std()

    return row


def collect_epoch_metrics(run_root: Path, init_type: str) -> pd.DataFrame:
    rows = []

    for epoch_metrics_path in sorted(run_root.glob("*/epoch_metrics.csv")):
        df = pd.read_csv(epoch_metrics_path)
        df["init_type"] = init_type
        df["case_dir"] = epoch_metrics_path.parent.name
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    metric_cols = [
        col
        for col in [
            "OccNet chamfer-L1",
            "OccNet chamfer-L2",
            "OccNet normal consistency",
            "OccNet f-scores",
            "VIoU",
            "Dice",
            "final_loss",
        ]
        if col in df.columns
    ]

    grouped = df.groupby(["init_type", "epoch"], as_index=False)
    summary = grouped[metric_cols].agg(["mean", "std", "count"])
    summary.columns = [
        "_".join([part for part in col if part])
        for col in summary.columns.to_flat_index()
    ]

    return summary.sort_values(["epoch", "init_type"])


def main():
    parser = argparse.ArgumentParser(
        description="Run meta-init and random-init validation back to back, then summarize results."
    )

    parser.add_argument(
        "--base_config",
        default=None,
        help="Optional base train_batch config. If omitted, --meta_config is used to build one.",
    )
    parser.add_argument(
        "--meta_config",
        default=None,
        help="config_used.yaml from the meta run. Used to auto-build the validation config.",
    )
    parser.add_argument(
        "--meta_checkpoint",
        required=True,
        help="Meta checkpoint to evaluate.",
    )
    parser.add_argument(
        "--comparison_name",
        required=True,
        help="Name for generated configs and output folder.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Data split to evaluate. Default: val.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--checkpoint_epochs",
        type=parse_int_list,
        default=parse_int_list("1,2,5,10,25,50,100"),
        help="Comma-separated epochs for intermediate evaluation.",
    )
    parser.add_argument("--grid_res", type=int, default=256)
    parser.add_argument("--grid_min", type=float, default=-0.5)
    parser.add_argument("--grid_max", type=float, default=0.5)
    parser.add_argument("--grid_batch_size", type=int, default=200000)
    parser.add_argument("--n_cube", type=int, default=256)
    parser.add_argument("--n_pointcloud", type=int, default=100000)
    parser.add_argument("--hash_resolution", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of cases for quick debugging.",
    )

    args = parser.parse_args()

    if args.base_config is not None:
        base_config = load_yaml(REPO_ROOT / args.base_config)
    elif args.meta_config is not None:
        meta_config = load_yaml(REPO_ROOT / args.meta_config)
        base_config = build_base_config_from_meta_config(meta_config, args)
    else:
        raise ValueError("Provide either --base_config or --meta_config.")

    comparison_root = REPO_ROOT / "outputs" / "meta_vs_random_comparison" / args.comparison_name
    generated_config_root = (
        REPO_ROOT / "outputs" / "generated_configs" / "meta_vs_random_comparison" / args.comparison_name
    )

    run_roots = {}

    for init_type in ["meta", "random"]:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("training", {})
        cfg.setdefault("data", {})
        cfg.setdefault("output", {})

        cfg["data"]["split"] = args.split
        cfg["training"]["init_type"] = init_type

        if init_type == "meta":
            cfg["training"]["init_checkpoint"] = args.meta_checkpoint
        else:
            cfg["training"]["init_checkpoint"] = None

        run_name = f"{args.comparison_name}_{init_type}"
        run_root = comparison_root / init_type
        run_roots[init_type] = run_root

        cfg["output"]["run_name"] = run_name
        cfg["output"]["output_root"] = str(run_root)

        generated_config_path = generated_config_root / f"{init_type}.yaml"
        save_yaml(cfg, generated_config_path)

        cmd = [
            sys.executable,
            "scripts/train_batch.py",
            "--config",
            str(generated_config_path),
            "--device",
            args.device,
        ]

        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        run_command(cmd)

    summary_rows = []
    for init_type, run_root in run_roots.items():
        row = summarize_run(run_root / "summary.csv", init_type)
        if row is not None:
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_out = comparison_root / "summary_comparison.csv"
    summary_df.to_csv(summary_out, index=False)

    epoch_dfs = [
        collect_epoch_metrics(run_root, init_type)
        for init_type, run_root in run_roots.items()
    ]
    epoch_df = pd.concat([df for df in epoch_dfs if not df.empty], ignore_index=True)

    if not epoch_df.empty:
        all_epoch_out = comparison_root / "epoch_metrics_all.csv"
        epoch_summary_out = comparison_root / "epoch_metrics_summary.csv"

        epoch_df.to_csv(all_epoch_out, index=False)
        summarize_epoch_metrics(epoch_df).to_csv(epoch_summary_out, index=False)

        print("\nSaved epoch metrics to:", all_epoch_out)
        print("Saved epoch summary to:", epoch_summary_out)
    else:
        print("\nNo epoch_metrics.csv files found to summarize.")

    print("\nSaved summary comparison to:", summary_out)
    print(summary_df)


if __name__ == "__main__":
    main()
