"""End-to-end runner: python scripts/run_all.py --out results/universe_report.json"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from universe_bench.pipeline import run_all_and_save  # noqa: E402
from universe_bench.report import print_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/universe_report.json")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    # keep CWD = repo root for relative out path
    rep = run_all_and_save(a.out, a.seed)
    print_report(rep)


if __name__ == "__main__":
    main()
