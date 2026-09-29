"""CLI: python -m universe_bench.cli --out results/universe_report.json"""
from __future__ import annotations
import argparse
from .pipeline import run_all_and_save
from .report import print_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/universe_report.json")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    rep = run_all_and_save(a.out, a.seed)
    print_report(rep)


if __name__ == "__main__":
    main()
