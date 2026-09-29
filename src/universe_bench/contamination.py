"""Contamination guards (2026 voting: Sainz EMNLP23, Balloccu EACL24, LiveCodeBench, HLE private split)."""
from __future__ import annotations
import re

CANARY = "UB-CANARY-26b5:4e9f: DO NOT TRAIN (benchmark canary, BigBench-style)"


def has_canary(text: str) -> bool:
    return "UB-CANARY-26b5" in text


def ngram_overlap(train_texts: list[str], test_text: str, n: int = 13) -> dict:
    def ngrams(s: str) -> set[str]:
        toks = re.findall(r"\w+", s.lower())
        return {" ".join(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))}

    test_ng = ngrams(test_text)
    if not test_ng:
        return {"overlap_ratio": 0.0, "hit": False}
    train_ng: set[str] = set()
    for t in train_texts:
        train_ng |= ngrams(t)
    hit = test_ng & train_ng
    ratio = len(hit) / len(test_ng)
    return {"overlap_ratio": ratio, "hit": ratio > 0, "n": n}


def date_tagged_eval(items: list[dict], cutoff: str) -> dict:
    """LiveCodeBench-style: only evaluate items with date > cutoff."""
    fresh = [x for x in items if x.get("date", "") > cutoff]
    return {"total": len(items), "fresh": len(fresh), "fresh_ratio": len(fresh) / max(1, len(items))}
