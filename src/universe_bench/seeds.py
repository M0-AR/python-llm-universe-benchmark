"""Global determinism: one place to seed everything."""
from __future__ import annotations
import os
import random


def seed_all(seed: int = 1337) -> int:
    seed = int(os.environ.get("UB_SEED", seed))
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "0")
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    return seed
