import numpy as np

from src.evaluation import mesh_utils


def fixed_OccNet_CD(
    pred_pointcloud,
    gt_pointcloud,
    pred_normals=None,
    gt_normals=None,
    Fscore_thresholds=np.linspace(1.0 / 1000, 1, 1000),
):
    """
    Same idea as mesh_utils.OccNet_CD, but computes F-score from per-point
    distances instead of scalar mean distances.
    """
    completeness_dist, completeness_normals = mesh_utils.distance_p2p(
        points_src=gt_pointcloud,
        normals_src=gt_normals,
        points_tgt=pred_pointcloud,
        normals_tgt=pred_normals,
    )

    completeness = completeness_dist.mean()
    completeness2 = (completeness_dist ** 2).mean()
    completeness_normals = completeness_normals.mean()

    accuracy_dist, accuracy_normals = mesh_utils.distance_p2p(
        points_src=pred_pointcloud,
        normals_src=pred_normals,
        points_tgt=gt_pointcloud,
        normals_tgt=gt_normals,
    )

    accuracy = accuracy_dist.mean()
    accuracy2 = (accuracy_dist ** 2).mean()
    accuracy_normals = accuracy_normals.mean()

    chamferL1 = 0.5 * (completeness + accuracy)
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = 0.5 * completeness_normals + 0.5 * accuracy_normals

    recall = mesh_utils.get_threshold_percentage(completeness_dist, Fscore_thresholds)
    precision = mesh_utils.get_threshold_percentage(accuracy_dist, Fscore_thresholds)

    F_scores = []
    for p, r in zip(precision, recall):
        denom = p + r
        F_scores.append(0.0 if denom == 0 else 2 * p * r / denom)

    return {
        "completeness": completeness,
        "accuracy": accuracy,
        "chamfer-L1": chamferL1,
        "completeness2": completeness2,
        "accuracy2": accuracy2,
        "chamfer-L2": chamferL2,
        "normals completeness": completeness_normals,
        "normals accuracy": accuracy_normals,
        "normal consistency": normals_correctness,
        "f-scores": F_scores,
    }


def apply_fscore_patch():
    """
    Patch mesh_utils.OccNet_CD in memory.
    """
    mesh_utils.OccNet_CD = fixed_OccNet_CD