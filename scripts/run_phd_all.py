"""Run all PhD chapters: python scripts/run_phd_all.py --out results/phd_report.json --md results/phd_report.md"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from universe_bench.phd_chapters import run_all_phd  # noqa: E402


def to_markdown(rep: dict) -> str:
    lines = ["# PhD Experiments Report (offline, seeded, frozen)", ""]
    for key in ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7"]:
        ch = rep.get(key, {})
        lines.append(f"## {key}: {ch.get('chapter','')}")
        lines.append(f"- claim: {ch.get('claim','')}")
        # chapter-specific one-liners
        if key == "ch1":
            lines.append(f"- acc base/upper/space/cs: {ch.get('acc_base')}/{ch.get('acc_upper')}/{ch.get('acc_trailing_space')}/{ch.get('acc_upper_case_sensitive')} masking={ch.get('masking_note','')[:160]}")
        if key == "ch2":
            s = ch.get("sequential", {})
            sc = ch.get("sequential_clear_win", {})
            lines.append(f"- sequential close: n={s.get('n_used')} {s.get('decision')} p={s.get('p')} save={s.get('savings_vs_200')} | clear: n={sc.get('n_used')} {sc.get('decision')} p={sc.get('p')} save={sc.get('savings_vs_200')}")
            lines.append(f"- irt ablation: {ch.get('irt_ablation')}")
        if key == "ch3":
            lines.append(f"- coverage_by_noise keys: {list(ch.get('coverage_by_noise', {}).keys())} rag_lift={ch.get('rag_lift')}")
        if key == "ch4":
            lines.append(f"- lost_in_middle={ch.get('lost_in_middle_scores')} noise_acc={ch.get('noise_accuracy_proxy')}")
        if key == "ch5":
            lines.append(f"- by_lang acc: en={ch.get('by_lang', {}).get('en')} de={ch.get('by_lang', {}).get('de')} ur={ch.get('by_lang', {}).get('ur_roman')} gap={ch.get('gap_en_minus_worst')}")
        if key == "ch6":
            lines.append(f"- kappa={ch.get('cohen_kappa')} win_rate={ch.get('win_rate_A')} bt={ch.get('bradley_terry')}")
        if key == "ch7":
            lines.append(f"- holdout={ch.get('holdout_n')} public={ch.get('public_n')} fresh={ch.get('fresh')} gates={ch.get('system', {}).get('gates')}")
        lines.append("")
    lines.append(f"versions: {rep.get('versions', {}).get('python','')[:60]}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phd_report.json")
    ap.add_argument("--md", default="results/phd_report.md")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()
    rep = run_all_phd(a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2))
    Path(a.md).write_text(to_markdown(rep))
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
