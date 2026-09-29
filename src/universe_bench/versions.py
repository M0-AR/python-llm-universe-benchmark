"""Versions map (Evalution-style): record everything needed to reproduce."""
from __future__ import annotations
import platform
import sys


def versions_map(extra: dict | None = None) -> dict:
    out: dict = {
        "python": sys.version,
        "platform": platform.platform(),
        "bench": "universe-0.1.0",
        "book": "Grigorov 2024 Intro to Python and LLMs (Apress)",
        "protocol": "frozen-prompts-v1 + seeded-splits-v1",
    }
    try:
        import numpy, sklearn, yaml  # type: ignore

        out.update(
            {
                "numpy": getattr(numpy, "__version__", None),
                "sklearn": getattr(sklearn, "__version__", None),
                "pyyaml": getattr(yaml, "__version__", None),
            }
        )
    except Exception as e:  # pragma: no cover
        out["deps_error"] = str(e)
    if extra:
        out.update(extra)
    return out
