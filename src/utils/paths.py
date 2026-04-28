from pathlib import Path

# Repository root:
# src/utils/paths.py -> src/utils -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Server workplace root
WORKPLACE_ROOT = Path("/home/bsc18/workplace")

# Dataset paths
DATASET_ROOT = (
    WORKPLACE_ROOT
    / "dataset"
    / "Totalsegmentator_dataset_v201_Liver"
    / "ok"
)

SAMPLED_ROOT = DATASET_ROOT / "sampled_20000"

NPY_ROOT = SAMPLED_ROOT / "npy" / "3D_Reconstruction"
MESH_ROOT = SAMPLED_ROOT / "mesh"

TRAIN_SPLIT = DATASET_ROOT / "train.txt"
VAL_SPLIT = DATASET_ROOT / "val.txt"
TEST_SPLIT = DATASET_ROOT / "test.txt"

# Repo-local outputs
OUTPUT_ROOT = REPO_ROOT / "outputs"
CONFIG_ROOT = REPO_ROOT / "configs"