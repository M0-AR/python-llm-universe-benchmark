"""Hallucination lifecycle (survey 2608.26168 + OpenHalDet 2606.06959, teaching version).

Taxonomy: factuality (wrong fact) vs faithfulness (unfaithful to source: intrinsic/extrinsic).
Stages: data-related / training-related / inference-related.
Detectors (offline, no LLM needed): lexical faithfulness (overlap with source), negation-flip heuristic,
self-consistency proxy (n-sample agreement given by caller).
"""
from __future__ import annotations
import re


def lexical_faithfulness(source: str, claim: str) -> dict:
    s = set(re.findall(r"\w+", source.lower()))
    c = re.findall(r"\w+", claim.lower())
    if not c:
        return {"coverage": 0.0, "verdict": "empty"}
    cov = sum(1 for w in c if w in s) / len(c)
    verdict = "faithful-ish" if cov >= 0.6 else ("extrinsic-risk" if cov >= 0.3 else "likely-hallucinated")
    return {"coverage": round(cov, 3), "verdict": verdict}


def classify_hallucination(gold_fact: str, source: str, output: str) -> dict:
    # intrinsic: contradicts source; extrinsic: adds unsupported; factuality: contradicts gold
    gf = gold_fact.lower()
    if gf and gf not in output.lower() and len(output.split()) > 2:
        fact = "factuality-risk"
    else:
        fact = "factuality-ok"
    faith = lexical_faithfulness(source, output)["verdict"]
    return {"factuality": fact, "faithfulness": faith}


def lifecycle_checklist() -> dict:
    return {
        "data": ["deduplicate (C4 50-word x60k lesson)", "decontaminate vs benchmarks", "filter toxicity/bias w/ human audit", "strip PII", "document machine-translated/OCR spans"],
        "training": ["record seeds/hparams", "versions map", "calibration (ECE/Brier)", "reward-model audit"],
        "inference": ["freeze prompts/decoding", "self-consistency n>=3 on risky Qs", "RAG grounding + citations", "human spot-check >=5%"],
    }
