"""Markdown + console reporting."""
from __future__ import annotations
import json


def to_markdown(report: dict) -> str:
    b = report.get("book", {})
    h = report.get("hidden_patterns", {}).get("token_bias", {})
    c = report.get("cost_latency", {}).get("synthetic_latency", {})
    lines = [
        "# Universe Benchmark Report (offline, reproducible)",
        "",
        f"- book_nlp_basics accuracy: {b.get('accuracy')} macro_f1: {b.get('macro_f1')} ci95: {b.get('acc_ci95')}",
        f"- preds: {b.get('preds')} gold: {b.get('gold')}",
        f"- token case bias: {h.get('case', {})}",
        f"- token numeric bias: {h.get('numeric', {})}",
        f"- token whitespace: {h.get('whitespace', {})}",
        f"- latency p50/p95: {c.get('p50')}/{c.get('p95')} n={c.get('n')}",
        f"- versions: {report.get('versions', {}).get('python', '')[:60]} | {report.get('versions', {}).get('numpy')}/{report.get('versions', {}).get('sklearn')}",
        "",
        "All prompts/splits/seeds frozen. See docs/BEST_PRACTICES_2026_2027.md",
    ]
    return "\n".join(lines)


def print_report(report: dict) -> None:
    print(to_markdown(report))
