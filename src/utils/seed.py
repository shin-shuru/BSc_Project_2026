import os
import random

import numpy as np
import torch


def seed_all(seed: int | None = None, logger=None):
    """
    Set random seed for reproducibility.
    """
    if seed is None:
        seed = 2024

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    numpy_generator = np.random.default_rng(seed=seed)
    torch_generator = torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if logger is None:
        print(f"Using random seed: {seed}")
    else:
        logger.info(f"Using random seed: {seed}")

    return torch_generator, numpy_generator