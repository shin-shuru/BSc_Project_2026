# scripts/plot_mesh_checkpoints.py

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def normalize_case_id(name: str) -> str:
    stem = Path(name).stem

    suffixes = [
        "_Segment_1",
        "_segment_1",
        "_mesh",
        "_predicted_mesh",
    ]

    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


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


def find_case_name(df: pd.DataFrame, case_query: str | None) -> str:
    cases = sorted(df["case_name"].unique())

    if len(cases) == 0:
        raise ValueError("No cases found in summary.csv")

    if case_query is None:
        return cases[0]

    query_norm = normalize_case_id(case_query)

    if query_norm in cases:
        return query_norm

    matches = [case for case in cases if query_norm in case or case in query_norm]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(
            f"Case query matched multiple cases: {matches}\n"
            f"Use the exact case_name."
        )

    raise ValueError(
        f"Could not find case matching: {case_query}\n"
        f"Available examples: {cases[:10]}"
    )


def find_pred_mesh(
    output_root: Path,
    init_type: str,
    case_name: str,
    epoch: int,
) -> Path:
    mesh_path = (
        output_root
        / init_type
        / case_name
        / f"epoch_{epoch:04d}"
        / "pred_mesh.stl"
    )

    if not mesh_path.exists():
        raise FileNotFoundError(f"Predicted mesh not found: {mesh_path}")

    return mesh_path


def find_gt_mesh_path(df_case: pd.DataFrame) -> Path | None:
    if "gt_mesh_path" not in df_case.columns:
        return None

    valid_paths = df_case["gt_mesh_path"].dropna().unique()

    if len(valid_paths) == 0:
        return None

    gt_path = Path(valid_paths[0])

    if gt_path.exists():
        return gt_path

    return None


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values()]
        )

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Could not load as trimesh.Trimesh: {mesh_path}")

    return mesh


def simplify_mesh_for_plot(mesh: trimesh.Trimesh, max_faces: int = 12000) -> trimesh.Trimesh:
    """
    Matplotlib 3D plotting becomes slow if the mesh has too many faces.
    This function keeps plotting manageable by sampling a subset of faces.

    It does not modify saved evaluation meshes; this is only for visualization.
    """
    if len(mesh.faces) <= max_faces:
        return mesh

    rng = np.random.default_rng(2024)
    chosen = rng.choice(len(mesh.faces), size=max_faces, replace=False)

    faces = mesh.faces[chosen]
    used_vertices = np.unique(faces.reshape(-1))

    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}

    new_vertices = mesh.vertices[used_vertices]

    remapped_faces = np.vectorize(vertex_map.get)(faces)

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=remapped_faces,
        process=False,
    )


def set_axes_equal(ax, bounds: np.ndarray):
    """
    Make 3D axes use equal scale so the liver is not visually distorted.
    """
    mins = bounds[0]
    maxs = bounds[1]

    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_mesh_on_axis(
    ax,
    mesh: trimesh.Trimesh,
    title: str,
    global_bounds: np.ndarray,
    elev: float,
    azim: float,
):
    mesh = simplify_mesh_for_plot(mesh)

    vertices = mesh.vertices
    faces = mesh.faces

    triangles = vertices[faces]

    collection = Poly3DCollection(
        triangles,
        linewidths=0.05,
        alpha=0.92,
    )

    collection.set_facecolor((0.72, 0.72, 0.72, 1.0))
    collection.set_edgecolor((0.18, 0.18, 0.18, 0.08))

    ax.add_collection3d(collection)

    set_axes_equal(ax, global_bounds)

    ax.view_init(elev=elev, azim=azim)

    ax.set_title(title, fontsize=10)

    ax.set_axis_off()


def collect_mesh_entries(
    *,
    df: pd.DataFrame,
    output_root: Path,
    case_name: str,
    meta_epochs: list[int],
    include_gt: bool,
    include_random: bool,
):
    df_case = df[df["case_name"] == case_name].copy()

    if len(df_case) == 0:
        raise ValueError(f"No rows found for case: {case_name}")

    entries: list[tuple[str, Path]] = []

    if include_gt:
        gt_path = find_gt_mesh_path(df_case)

        if gt_path is not None:
            entries.append(("Ground Truth", gt_path))
        else:
            print("[WARN] Could not find valid ground-truth mesh path in summary.csv")

    if include_random:
        random_rows = df_case[df_case["init_type"] == "random"].copy()

        if len(random_rows) > 0:
            random_epoch = int(random_rows["eval_epoch"].max())
            random_mesh_path = find_pred_mesh(
                output_root=output_root,
                init_type="random",
                case_name=case_name,
                epoch=random_epoch,
            )
            entries.append((f"Random\nEpoch {random_epoch}", random_mesh_path))
        else:
            print("[WARN] No random rows found for this case")

    meta_rows = df_case[df_case["init_type"] == "meta"].copy()

    if len(meta_rows) == 0:
        raise ValueError(f"No meta rows found for case: {case_name}")

    available_meta_epochs = sorted(meta_rows["eval_epoch"].unique())

    if len(meta_epochs) == 0:
        meta_epochs = available_meta_epochs

    for epoch in meta_epochs:
        if epoch not in available_meta_epochs:
            print(f"[SKIP] Meta epoch {epoch} not available for case {case_name}")
            continue

        meta_mesh_path = find_pred_mesh(
            output_root=output_root,
            init_type="meta",
            case_name=case_name,
            epoch=epoch,
        )

        entries.append((f"Meta\nEpoch {epoch}", meta_mesh_path))

    if len(entries) == 0:
        raise ValueError("No mesh entries collected for plotting")

    return entries


def compute_global_bounds(meshes: list[trimesh.Trimesh]) -> np.ndarray:
    mins = []
    maxs = []

    for mesh in meshes:
        bounds = mesh.bounds
        mins.append(bounds[0])
        maxs.append(bounds[1])

    global_min = np.min(np.stack(mins, axis=0), axis=0)
    global_max = np.max(np.stack(maxs, axis=0), axis=0)

    return np.stack([global_min, global_max], axis=0)


def plot_mesh_grid(
    *,
    entries: list[tuple[str, Path]],
    out_path: Path,
    ncols: int,
    elev: float,
    azim: float,
    dpi: int,
):
    loaded = []

    print("Loading meshes:")
    for title, path in entries:
        print(f"  {title.replace(chr(10), ' ')}: {path}")
        loaded.append((title, load_mesh(path)))

    meshes = [mesh for _, mesh in loaded]
    global_bounds = compute_global_bounds(meshes)

    n = len(loaded)
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(3.1 * ncols, 3.3 * nrows))

    for idx, (title, mesh) in enumerate(loaded, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")

        plot_mesh_on_axis(
            ax=ax,
            mesh=mesh,
            title=title,
            global_bounds=global_bounds,
            elev=elev,
            azim=azim,
        )

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side mesh checkpoint visualizations from "
            "the meta-vs-random evaluation output."
        )
    )

    parser.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="Path to summary.csv produced by scripts/evaluate_meta_init.py",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Evaluation output root. Defaults to the parent directory of summary_csv."
        ),
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help=(
            "Case name to plot, e.g. s0004. "
            "If omitted, the first case in summary.csv is used."
        ),
    )

    parser.add_argument(
        "--meta_epochs",
        nargs="*",
        type=int,
        default=[0, 1, 2, 5, 10, 20, 50, 100, 300],
        help="Meta checkpoint epochs to include in the visual grid.",
    )

    parser.add_argument(
        "--no_gt",
        action="store_true",
        help="Do not include ground-truth mesh.",
    )

    parser.add_argument(
        "--no_random",
        action="store_true",
        help="Do not include random final mesh.",
    )

    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in the output image grid.",
    )

    parser.add_argument(
        "--elev",
        type=float,
        default=20.0,
        help="3D camera elevation angle.",
    )

    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="3D camera azimuth angle.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path. Defaults to <output_root>/mesh_figures/<case>_checkpoints.png",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output image DPI.",
    )

    args = parser.parse_args()

    summary_csv = resolve_path(args.summary_csv)
    df = load_summary(summary_csv)

    output_root = (
        resolve_path(args.output_root)
        if args.output_root is not None
        else summary_csv.parent
    )

    case_name = find_case_name(df, args.case)

    entries = collect_mesh_entries(
        df=df,
        output_root=output_root,
        case_name=case_name,
        meta_epochs=args.meta_epochs,
        include_gt=not args.no_gt,
        include_random=not args.no_random,
    )

    if args.out is None:
        out_path = output_root / "mesh_figures" / f"{case_name}_checkpoints.png"
    else:
        out_path = resolve_path(args.out)

    plot_mesh_grid(
        entries=entries,
        out_path=out_path,
        ncols=args.ncols,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()