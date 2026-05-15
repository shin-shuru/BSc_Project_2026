import argparse
import copy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
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
            "epochs": args.meta_epochs,
            "checkpoint_epochs": args.meta_checkpoint_epochs,
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


def run_command(cmd: list[str]) -> float:
    print("\nRunning:")
    print(" ".join(cmd))

    start_time = time.perf_counter()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed_time_sec = time.perf_counter() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")

    return elapsed_time_sec


def summarize_run(
    summary_path: Path,
    init_type: str,
    elapsed_time_sec: float | None = None,
    configured_epochs: int | None = None,
) -> dict | None:
    if not summary_path.exists():
        return None

    df = pd.read_csv(summary_path)
    df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()

    if len(df_ok) == 0:
        return {
            "init_type": init_type,
            "n": 0,
            "status": "no_ok_cases",
            "elapsed_time_sec": elapsed_time_sec,
            "configured_epochs": configured_epochs,
        }

    row = {
        "init_type": init_type,
        "n": len(df_ok),
        "status": "ok",
        "elapsed_time_sec": elapsed_time_sec,
        "elapsed_time_min": elapsed_time_sec / 60 if elapsed_time_sec is not None else None,
        "configured_epochs": configured_epochs,
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


def render_mesh_grid_for_case(
    run_root: Path,
    init_type: str,
    case_name: str,
    epochs: list[int],
    output_dir: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    mesh_paths = [
        run_root / case_name / "epoch_evaluations" / f"epoch_{epoch:04d}" / "pred_mesh.stl"
        for epoch in epochs
    ]
    existing = [(epoch, path) for epoch, path in zip(epochs, mesh_paths) if path.exists()]

    if not existing:
        return

    n_cols = min(4, len(existing))
    n_rows = int(np.ceil(len(existing) / n_cols))
    fig = plt.figure(figsize=(4 * n_cols, 3.8 * n_rows))

    for idx, (epoch, mesh_path) in enumerate(existing, start=1):
        ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
        mesh = trimesh.load_mesh(mesh_path, process=False)

        if len(mesh.vertices) == 0:
            ax.set_title(f"epoch {epoch}\nempty mesh")
            ax.axis("off")
            continue

        points = mesh.vertices
        if len(points) > 6000:
            rng = np.random.default_rng(2024)
            points = points[rng.choice(len(points), size=6000, replace=False)]

        center = points.mean(axis=0)
        points = points - center
        scale = np.abs(points).max()
        if scale > 0:
            points = points / scale

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c="#2f6f9f", alpha=0.75)
        ax.view_init(elev=22, azim=-58)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"epoch {epoch}", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"{init_type} | {case_name}", fontsize=14)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{init_type}_{case_name}_mesh_grid.png", dpi=180)
    plt.close(fig)


def make_metric_threshold_plots(epoch_df: pd.DataFrame, output_dir: Path):
    if epoch_df.empty:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("VIoU", "max", "VIoU"),
        ("OccNet chamfer-L1", "min", "Chamfer L1"),
        ("OccNet f-scores", "max", "F1-score"),
    ]

    for metric_col, best_mode, label in metrics:
        if metric_col not in epoch_df.columns:
            continue

        random_df = epoch_df[epoch_df["init_type"] == "random"]
        meta_df = epoch_df[epoch_df["init_type"] == "meta"]

        if random_df.empty or meta_df.empty:
            continue

        random_by_epoch = random_df.groupby("epoch")[metric_col].mean().sort_index()
        meta_by_epoch = meta_df.groupby("epoch")[metric_col].mean().sort_index()

        threshold = random_by_epoch.max() if best_mode == "max" else random_by_epoch.min()

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            meta_by_epoch.index,
            meta_by_epoch.values,
            marker="o",
            linewidth=2,
            label="Meta mean by epoch",
        )
        ax.axhline(
            threshold,
            color="#b33a3a",
            linestyle="--",
            linewidth=1.8,
            label=f"Best random threshold ({threshold:.6g})",
        )
        ax.set_xlabel("Adaptation epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Meta {label} vs best random threshold")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()

        safe_name = label.lower().replace(" ", "_").replace("-", "_")
        fig.savefig(output_dir / f"meta_vs_random_threshold_{safe_name}.png", dpi=180)
        plt.close(fig)


def make_visualizations(comparison_root: Path, run_roots: dict[str, Path], epoch_df: pd.DataFrame):
    visual_dir = comparison_root / "visualizations"

    try:
        make_metric_threshold_plots(epoch_df, visual_dir)

        if epoch_df.empty or "case_dir" not in epoch_df.columns or "epoch" not in epoch_df.columns:
            return

        epochs_by_init = {
            init_type: sorted(
                int(epoch)
                for epoch in epoch_df[epoch_df["init_type"] == init_type]["epoch"].dropna().unique()
            )
            for init_type in ["meta", "random"]
        }
        case_names = sorted(epoch_df["case_dir"].dropna().unique())

        for case_name in case_names:
            for init_type, run_root in run_roots.items():
                render_mesh_grid_for_case(
                    run_root=run_root,
                    init_type=init_type,
                    case_name=case_name,
                    epochs=epochs_by_init.get(init_type, []),
                    output_dir=visual_dir,
                )

        print("Saved visualizations to:", visual_dir)
    except Exception as exc:
        print(f"[WARN] Visualization generation failed: {exc}")


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--meta_epochs",
        type=int,
        default=200,
        help="Epochs for meta-init adaptation. Defaults to --epochs.",
    )
    parser.add_argument(
        "--random_epochs",
        type=int,
        default=600,
        help="Epochs for random-init adaptation. Defaults to --epochs.",
    )
    parser.add_argument(
        "--meta_checkpoint_epochs",
        type=parse_int_list,
        default=parse_int_list("1,2,5,10,25,50,100,200"),
        help="Comma-separated intermediate evaluation epochs for meta-init. Defaults to --checkpoint_epochs.",
    )
    parser.add_argument(
        "--random_checkpoint_epochs",
        type=parse_int_list,
        default=parse_int_list("1,2,5,10,25,50,100,200,400,600"),
        help="Comma-separated intermediate evaluation epochs for random-init. Defaults to --checkpoint_epochs.",
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
    args.meta_epochs = args.meta_epochs
    args.random_epochs = args.random_epochs
    args.meta_checkpoint_epochs = args.meta_checkpoint_epochs
    args.random_checkpoint_epochs = args.random_checkpoint_epochs

    if args.base_config is not None:
        base_config = load_yaml(REPO_ROOT / args.base_config)
    elif args.meta_config is not None:
        meta_config = load_yaml(REPO_ROOT / args.meta_config)
        base_config = build_base_config_from_meta_config(meta_config, args)
    else:
        raise ValueError("Provide either --base_config or --meta_config.")

    comparison_root = REPO_ROOT / "outputs" / "meta_vs_random_comparison2" / args.comparison_name
    generated_config_root = (
        REPO_ROOT / "outputs" / "generated_configs" / "meta_vs_random_comparison2" / args.comparison_name
    )

    run_roots = {}
    elapsed_times = {}
    configured_epochs = {}

    for init_type in ["meta", "random"]:
        cfg = copy.deepcopy(base_config)
        cfg.setdefault("training", {})
        cfg.setdefault("data", {})
        cfg.setdefault("output", {})

        cfg["data"]["split"] = args.split
        cfg["training"]["init_type"] = init_type
        cfg["training"]["epochs"] = (
            args.meta_epochs if init_type == "meta" else args.random_epochs
        )
        cfg["training"]["checkpoint_epochs"] = (
            args.meta_checkpoint_epochs
            if init_type == "meta"
            else args.random_checkpoint_epochs
        )
        configured_epochs[init_type] = cfg["training"]["epochs"]

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

        elapsed_times[init_type] = run_command(cmd)

    summary_rows = []
    for init_type, run_root in run_roots.items():
        row = summarize_run(
            run_root / "summary.csv",
            init_type,
            elapsed_time_sec=elapsed_times.get(init_type),
            configured_epochs=configured_epochs.get(init_type),
        )
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
        make_visualizations(comparison_root, run_roots, epoch_df)

        print("\nSaved epoch metrics to:", all_epoch_out)
        print("Saved epoch summary to:", epoch_summary_out)
    else:
        print("\nNo epoch_metrics.csv files found to summarize.")

    print("\nSaved summary comparison to:", summary_out)
    print(summary_df)


if __name__ == "__main__":
    main()
