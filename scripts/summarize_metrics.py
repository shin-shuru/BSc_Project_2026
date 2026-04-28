import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "OccNet chamfer-L1",
    "VIoU",
    "OccNet f-scores",
    "final_loss",
]


def main():
    parser = argparse.ArgumentParser(
        description="Summarize INR metrics from summary.csv."
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="Path to summary.csv.",
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_csv)
    df = pd.read_csv(summary_path)

    df_ok = df[df["status"] == "ok"].copy()

    rows = []

    for col in METRIC_COLUMNS:
        if col not in df_ok.columns:
            continue

        rows.append({
            "metric": col,
            "mean": df_ok[col].mean(),
            "std": df_ok[col].std(),
            "n": df_ok[col].count(),
        })

    out_df = pd.DataFrame(rows)

    out_path = summary_path.parent / "metric_summary.csv"
    out_df.to_csv(out_path, index=False)

    print(out_df)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()