# scripts/plot_meta_vs_random.py
"""
python scripts/plot_meta_vs_random.py --summary_csv outputs/meta_vs_random_comparison/siren_meta_vs_random_single_run/summary.csv
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]


METRICS = {
    "OccNet chamfer-L1": {
        "safe_name": "chamfer_l1",
        "label": "OccNet Chamfer-L1",
        "direction": "lower",
        "ylabel": "Chamfer-L1 ↓",
    },
    "VIoU": {
        "safe_name": "viou",
        "label": "Volume IoU",
        "direction": "higher",
        "ylabel": "VIoU ↑",
    },
    "OccNet f-scores": {
        "safe_name": "fscore",
        "label": "F-score",
        "direction": "higher",
        "ylabel": "F-score ↑",
    },
}


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_summary(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")

    df = pd.read_csv(summary_csv)

    required_cols = [
        "case_name",
        "init_type",
        "eval_epoch",
        "status",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"summary.csv is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[df["status"] == "ok"].copy()

    if len(df) == 0:
        raise ValueError("No successful rows found in summary.csv")

    df["eval_epoch"] = pd.to_numeric(df["eval_epoch"], errors="coerce")
    df = df.dropna(subset=["eval_epoch"])
    df["eval_epoch"] = df["eval_epoch"].astype(int)

    return df


def get_random_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each case, select the final random-init row.
    This works even if random was evaluated at more than one epoch.
    """
    random_df = df[df["init_type"] == "random"].copy()

    if len(random_df) == 0:
        raise ValueError("No random-init rows found in summary.csv")

    random_final = (
        random_df.sort_values("eval_epoch")
        .groupby("case_name", as_index=False)
        .tail(1)
        .copy()
    )

    return random_final


def summarize_meta_by_epoch(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    meta_df = df[df["init_type"] == "meta"].copy()

    if len(meta_df) == 0:
        raise ValueError("No meta-init rows found in summary.csv")

    if metric_col not in meta_df.columns:
        raise ValueError(f"Metric column not found: {metric_col}")

    meta_df[metric_col] = pd.to_numeric(meta_df[metric_col], errors="coerce")
    meta_df = meta_df.dropna(subset=[metric_col])

    grouped = (
        meta_df.groupby("eval_epoch")[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
        .sort_values("eval_epoch")
    )

    grouped["std"] = grouped["std"].fillna(0.0)

    return grouped


def summarize_random_final(df: pd.DataFrame, metric_col: str) -> dict:
    random_final = get_random_final_rows(df)

    if metric_col not in random_final.columns:
        raise ValueError(f"Metric column not found: {metric_col}")

    values = pd.to_numeric(random_final[metric_col], errors="coerce").dropna()

    if len(values) == 0:
        raise ValueError(f"No valid random-final values found for {metric_col}")

    return {
        "mean": float(values.mean()),
        "std": float(values.std()) if len(values) > 1 else 0.0,
        "n": int(values.count()),
        "epoch": int(random_final["eval_epoch"].max()),
    }


def find_first_epoch_reaching_baseline(
    meta_summary: pd.DataFrame,
    random_mean: float,
    direction: str,
) -> int | None:
    """
    Find the first meta epoch where the mean meta result reaches or beats
    the mean random-final baseline.

    lower: meta <= random baseline
    higher: meta >= random baseline
    """
    for _, row in meta_summary.iterrows():
        epoch = int(row["eval_epoch"])
        meta_mean = float(row["mean"])

        if direction == "lower" and meta_mean <= random_mean:
            return epoch

        if direction == "higher" and meta_mean >= random_mean:
            return epoch

    return None


def plot_metric(
    *,
    df: pd.DataFrame,
    metric_col: str,
    metric_info: dict,
    out_dir: Path,
    use_log_x: bool,
    show_std: bool,
    dpi: int,
):
    meta_summary = summarize_meta_by_epoch(df, metric_col)
    random_summary = summarize_random_final(df, metric_col)

    random_mean = random_summary["mean"]
    random_std = random_summary["std"]
    random_epoch = random_summary["epoch"]

    first_reach_epoch = find_first_epoch_reaching_baseline(
        meta_summary=meta_summary,
        random_mean=random_mean,
        direction=metric_info["direction"],
    )

    x = meta_summary["eval_epoch"].to_numpy()
    y = meta_summary["mean"].to_numpy()
    y_std = meta_summary["std"].to_numpy()

    # For log-scale x-axis, epoch 0 cannot be plotted directly.
    # Plot epoch 0 at x=0 if linear, or shift it to x=0.5 if log.
    if use_log_x:
        x_plot = np.where(x == 0, 0.5, x)
        baseline_x_min = max(0.5, float(x_plot.min()))
    else:
        x_plot = x
        baseline_x_min = float(x_plot.min())

    baseline_x_max = float(x_plot.max())

    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Random final baseline.
    ax.hlines(
        y=random_mean,
        xmin=baseline_x_min,
        xmax=baseline_x_max,
        colors="tab:blue",
        linewidth=2.5,
        label=f"Random INR final mean, epoch {random_epoch}",
    )

    if show_std and random_std > 0:
        ax.fill_between(
            [baseline_x_min, baseline_x_max],
            [random_mean - random_std, random_mean - random_std],
            [random_mean + random_std, random_mean + random_std],
            color="tab:blue",
            alpha=0.12,
            label="Random final ±1 SD",
        )

    # Meta adaptation curve.
    ax.plot(
        x_plot,
        y,
        color="tab:red",
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="Meta-INR mean",
    )

    if show_std:
        ax.fill_between(
            x_plot,
            y - y_std,
            y + y_std,
            color="tab:red",
            alpha=0.16,
            label="Meta-INR ±1 SD",
        )

    # Mark first epoch where meta reaches random-final baseline.
    if first_reach_epoch is not None:
        reach_x = 0.5 if use_log_x and first_reach_epoch == 0 else first_reach_epoch

        ax.axvline(
            reach_x,
            linestyle="--",
            linewidth=1.5,
            color="gray",
            alpha=0.8,
        )

        ax.annotate(
            f"Meta reaches random final\nat epoch {first_reach_epoch}",
            xy=(reach_x, random_mean),
            xytext=(8, 18),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", linewidth=1.0),
        )

    title = (
        f"{metric_info['label']}: Meta-INR adaptation "
        f"vs random INR final baseline"
    )

    ax.set_title(title)
    ax.set_xlabel("Meta-INR adaptation epoch")
    ax.set_ylabel(metric_info["ylabel"])

    if use_log_x:
        ax.set_xscale("log")
        tick_positions = x_plot
        tick_labels = [str(int(v)) for v in x]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks(x_plot)
        ax.set_xticklabels([str(int(v)) for v in x])

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{metric_info['safe_name']}_meta_vs_random.png"
    pdf_path = out_dir / f"{metric_info['safe_name']}_meta_vs_random.pdf"

    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    return {
        "metric": metric_col,
        "random_final_epoch": random_epoch,
        "random_final_mean": random_mean,
        "random_final_std": random_std,
        "random_final_n": random_summary["n"],
        "first_meta_epoch_reaching_random_final": first_reach_epoch,
    }


def save_plot_summary(rows: list[dict], out_dir: Path):
    out_df = pd.DataFrame(rows)
    out_path = out_dir / "plot_summary.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot meta-INR checkpoint metrics against final random-INR baseline."
        )
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="Path to summary.csv produced by scripts/evaluate_meta_init.py",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to <summary_csv parent>/plots",
    )

    parser.add_argument(
        "--linear_x",
        action="store_true",
        help="Use linear x-axis instead of log x-axis.",
    )

    parser.add_argument(
        "--no_std",
        action="store_true",
        help="Disable ±1 standard deviation bands.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output.",
    )

    args = parser.parse_args()

    summary_csv = resolve_path(args.summary_csv)

    if args.out_dir is None:
        out_dir = summary_csv.parent / "plots"
    else:
        out_dir = resolve_path(args.out_dir)

    df = load_summary(summary_csv)

    rows = []

    for metric_col, metric_info in METRICS.items():
        if metric_col not in df.columns:
            print(f"[SKIP] Metric not found in summary.csv: {metric_col}")
            continue

        row = plot_metric(
            df=df,
            metric_col=metric_col,
            metric_info=metric_info,
            out_dir=out_dir,
            use_log_x=not args.linear_x,
            show_std=not args.no_std,
            dpi=args.dpi,
        )

        rows.append(row)

    save_plot_summary(rows, out_dir)


if __name__ == "__main__":
    main()