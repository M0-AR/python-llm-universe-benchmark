"""Frozen pipeline: prompts + seeds + splits locked; every run emits versions manifest."""
from __future__ import annotations
import json
import time
from pathlib import Path

from . import seeds as _seeds
from . import versions as _versions
from . import datasets_book as _data
from . import metrics as _m
from . import contamination as _contam
from . import tokenization_bias as _tok
from . import hallucination_lifecycle as _hallu
from . import cost_latency as _cost


def _tfidf_baseline_predict(train: list[tuple[str, str]], texts: list[str]) -> list[str]:
    # Minimal TF-IDF + centroid classifier (no heavy deps beyond sklearn)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    Xtr = [t for t, _ in train]
    ytr = [y for _, y in train]
    vec = TfidfVectorizer()
    X = vec.fit_transform(Xtr)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, ytr)
    return list(clf.predict(vec.transform(texts)))


def run_book_suite(seed: int = 1337) -> dict:
    seed = _seeds.seed_all(seed)
    t0 = time.perf_counter()
    split = _data.sentiment_split(seed)
    preds_raw = _tfidf_baseline_predict(split.train, [t for t, _ in split.test])
    preds = [str(x) for x in preds_raw]
    gold = [y for _, y in split.test]
    acc = _m.accuracy(gold, preds)
    f1 = _m.macro_f1(gold, preds)
    # bootstrap on 0/1 correctness
    corr = [1.0 if a == b else 0.0 for a, b in zip(gold, preds)]
    mean, lo, hi = _m.bootstrap_ci(corr, n_boot=500, seed=seed)
    latency = time.perf_counter() - t0
    return {
        "suite": "book_nlp_basics",
        "seed": seed,
        "frozen_prompts": _data.FROZEN_PROMPTS,
        "accuracy": acc,
        "macro_f1": f1,
        "acc_ci95": [lo, hi],
        "latency_s": latency,
        "preds": preds,
        "gold": gold,
    }


def run_hidden_patterns() -> dict:
    return {
        "token_bias": _tok.full_report(),
        "contamination_demo": {
            "canary_present": _contam.has_canary("eval " + _contam.CANARY),
            "overlap": _contam.ngram_overlap(
                ["ELIZA was created in 1966 by Joseph Weizenbaum at MIT."],
                "ELIZA was created in 1966 by Joseph Weizenbaum at MIT.",
                n=5,
            ),
            "date_tagged": _contam.date_tagged_eval(
                [{"id": 1, "date": "2026-01-01"}, {"id": 2, "date": "2024-01-01"}], cutoff="2025-01-01"
            ),
        },
        "hallucination_demo": {
            "faithful": _hallu.lexical_faithfulness(
                "ELIZA was created in 1966 by Joseph Weizenbaum at MIT.",
                "ELIZA was created in 1966 by Weizenbaum",
            ),
            "unfaithful": _hallu.lexical_faithfulness(
                "ELIZA was created in 1966 by Joseph Weizenbaum at MIT.",
                "ELIZA invented the moon landing in 1999 with quantum blockchain",
            ),
            "checklist": _hallu.lifecycle_checklist(),
        },
    }


def run_all_and_save(out_path: str = "results/universe_report.json", seed: int = 1337) -> dict:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    book = run_book_suite(seed)
    hidden = run_hidden_patterns()
    # cost demo: synthetic 1k req latencies
    import random

    rng = random.Random(seed)
    lat = [rng.gauss(0.08, 0.02) for _ in range(200)]
    lat = [max(0.01, x) for x in lat]
    report = {
        "versions": _versions.versions_map(),
        "book": book,
        "hidden_patterns": hidden,
        "cost_latency": {
            "synthetic_latency": _cost.summarize_runs(lat),
            "tokens_per_sec_demo": _cost.tokens_per_sec(1000, 0.08),
            "cost_per_1k_demo_usd": _cost.cost_per_1k(800, 200, 0.01, 0.03),
            "co2e_per_1m_kg_demo": _cost.co2e_per_1m_tokens_estimate(),
        },
        "reproducibility": [
            "PYTHONHASHSEED=0",
            "UB_SEED frozen",
            "prompts frozen v1",
            "splits frozen v1",
            "canary included",
            "versions map stored",
        ],
    }
    out.write_text(json.dumps(report, indent=2))
    return report
