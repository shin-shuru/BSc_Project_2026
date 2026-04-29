import argparse
import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


EPOCHS_TO_TEST = [10, 25, 50, 100, 200]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_command(cmd: list[str]):
    print("\nRunning:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def summarize_results(output_root: Path):
    rows = []

    for summary_path in sorted(output_root.rglob("summary.csv")):
        df = pd.read_csv(summary_path)
        df_ok = df[df["status"] == "ok"].copy()

        if len(df_ok) == 0:
            continue

        run_dir = summary_path.parent

        row = {
            "run_name": run_dir.name,
            "n": len(df_ok),
            "init_type": df_ok["init_type"].iloc[0] if "init_type" in df_ok.columns else "",
            "epochs": df_ok["epochs"].iloc[0] if "epochs" in df_ok.columns else "",
            "mean_chamfer_l1": df_ok["OccNet chamfer-L1"].mean(),
            "std_chamfer_l1": df_ok["OccNet chamfer-L1"].std(),
            "mean_viou": df_ok["VIoU"].mean(),
            "std_viou": df_ok["VIoU"].std(),
            "mean_fscore": df_ok["OccNet f-scores"].mean(),
            "std_fscore": df_ok["OccNet f-scores"].std(),
            "mean_final_loss": df_ok["final_loss"].mean(),
            "std_final_loss": df_ok["final_loss"].std(),
        }

        rows.append(row)

    out_df = pd.DataFrame(rows)

    if len(out_df) > 0:
        out_df = out_df.sort_values(["epochs", "init_type"])
        out_path = output_root / "comparison_summary.csv"
        out_df.to_csv(out_path, index=False)

        print("\nComparison summary:")
        print(out_df)
        print("\nSaved to:", out_path)
    else:
        print("No valid summary.csv files found.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare random SIREN init vs meta-learned SIREN init."
    )

    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/siren.yaml",
        help="Base SIREN config.",
    )

    parser.add_argument(
        "--meta_checkpoint",
        type=str,
        required=True,
        help="Path to meta_init.pt.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test cases for quick debugging.",
    )

    parser.add_argument(
        "--epochs",
        nargs="*",
        type=int,
        default=EPOCHS_TO_TEST,
        help="Epoch values to compare.",
    )

    args = parser.parse_args()

    base_config_path = REPO_ROOT / args.base_config
    base_config = load_yaml(base_config_path)

    output_root = REPO_ROOT / "outputs" / "meta_vs_random_comparison"
    config_root = REPO_ROOT / "outputs" / "generated_configs" / "meta_vs_random"

    for epochs in args.epochs:
        for init_type in ["random", "meta"]:
            cfg = copy.deepcopy(base_config)

            cfg["training"]["epochs"] = epochs
            cfg["training"]["init_type"] = init_type

            # Keep architecture identical to your meta-trained SIREN.
            cfg["model"]["first_w0"] = 15.0
            cfg["model"]["hidden_w0"] = 10.0

            if init_type == "meta":
                cfg["training"]["init_checkpoint"] = args.meta_checkpoint
            else:
                cfg["training"]["init_checkpoint"] = None

            run_name = f"siren_{init_type}_epochs_{epochs}"
            cfg["output"]["run_name"] = run_name
            cfg["output"]["output_root"] = str(output_root / run_name)

            generated_config_path = config_root / f"{run_name}.yaml"
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

    summarize_results(output_root)


if __name__ == "__main__":
    main()