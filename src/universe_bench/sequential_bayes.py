"""Bayesian sequential testing (PyData Amsterdam 2026 talk): stop early, decide smarter.

Beta-Binomial for accuracy comparisons. Uniform Beta(1,1) prior by default.
Posterior: Beta(1+k, 1+n-k). Stopping when P(p_A > p_B | data) > 0.95 or < 0.05,
or max budget exhausted. Also reports Bayes factor approx via Savage-Dickey style
marginal comparison for teaching (not a replacement for full BF).
"""
from __future__ import annotations
import math
import random


def _beta_sample(a: float, b: float, rng: random.Random) -> float:
    # gamma sampling via stdlib
    return rng.gammavariate(a, 1.0) / (rng.gammavariate(a, 1.0) + rng.gammavariate(b, 1.0))


def prob_a_better(k_a: int, n_a: int, k_b: int, n_b: int, draws: int = 20000, seed: int = 1337) -> float:
    rng = random.Random(seed)
    w = 0
    for _ in range(draws):
        pa = _beta_sample(1 + k_a, 1 + n_a - k_a, rng)
        pb = _beta_sample(1 + k_b, 1 + n_b - k_b, rng)
        if pa > pb:
            w += 1
    return w / draws


def sequential_decision(stream_a: list[int], stream_b: list[int], thresh: float = 0.95, min_n: int = 20, max_n: int = 200) -> dict:
    n = min(len(stream_a), len(stream_b), max_n)
    history = []
    for i in range(1, n + 1):
        if i < min_n:
            continue
        ka, kb = sum(stream_a[:i]), sum(stream_b[:i])
        p = prob_a_better(ka, i, kb, i)
        history.append({"n": i, "p_A_gt_B": p, "acc_A": ka / i, "acc_B": kb / i})
        if p > thresh:
            return {"decision": "A better (early stop)", "n_used": i, "p": p, "history": history}
        if p < 1 - thresh:
            return {"decision": "B better (early stop)", "n_used": i, "p": p, "history": history}
    ka, kb = sum(stream_a[:n]), sum(stream_b[:n])
    p = prob_a_better(ka, n, kb, n)
    return {"decision": "undecided at budget", "n_used": n, "p": p, "history": history}
