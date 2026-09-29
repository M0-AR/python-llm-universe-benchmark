"""Metrics with uncertainty (2026 checklist): accuracy/F1/EM + pass@k unbiased + ECE/Brier + win-rate bootstrap + McNemar."""
from __future__ import annotations
import math
import random
from collections import Counter


def accuracy(y_true: list, y_pred: list) -> float:
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def macro_f1(y_true: list, y_pred: list) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / max(1, len(f1s))


def exact_match(gold: str, pred: str) -> float:
    import string

    def norm(s: str) -> str:
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())

    return 1.0 if norm(gold) == norm(pred) else 0.0


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Wiki/Python-correct unbiased estimator: 1 - C(N-c, k)/C(N, k)."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 1337, alpha: float = 0.05) -> tuple[float, float, float]:
    rng = random.Random(seed)
    m = len(vals)
    mean = sum(vals) / m
    boots = []
    for _ in range(n_boot):
        s = [vals[rng.randrange(m)] for _ in range(m)]
        boots.append(sum(s) / m)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return mean, lo, hi


def mcnemar(b01: int, b10: int) -> dict:
    """McNemar for paired classifiers: b01=A wrong B right, b10=A right B wrong."""
    from math import erf, sqrt

    n = b01 + b10
    if n == 0:
        return {"stat": 0.0, "p": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / n  # continuity-corrected chi2(1)
    # chi2(1) survival via erf
    p = 1 - erf(math.sqrt(stat / 2))
    return {"stat": float(stat), "p": float(p), "note": "chi2(1) continuity-corrected"}


def ece_brier(y_true01: list[int], p_hat: list[float], n_bins: int = 10) -> dict:
    assert len(y_true01) == len(p_hat)
    brier = sum((p - y) ** 2 for p, y in zip(p_hat, y_true01)) / len(y_true01)
    bins: list[list[int]] = [[] for _ in range(n_bins)]
    pbins: list[list[float]] = [[] for _ in range(n_bins)]
    for y, p in zip(y_true01, p_hat):
        b = min(n_bins - 1, int(p * n_bins))
        bins[b].append(y)
        pbins[b].append(p)
    ece = 0.0
    for ys, ps in zip(bins, pbins):
        if not ys:
            continue
        acc = sum(ys) / len(ys)
        conf = sum(ps) / len(ps)
        ece += abs(acc - conf) * len(ys) / len(y_true01)
    return {"brier": brier, "ece": ece}
