"""Verify reproducibility: re-run twice, diff key fields (no hand-edit without verify)."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path


def run_once(out: str, seed: int = 1337) -> dict:
    subprocess.run([sys.executable, "scripts/run_all.py", "--out", out, "--seed", str(seed)], check=True, capture_output=True)
    return json.loads(Path(out).read_text())


def _strip_volatile(obj):
    # Drop wall-clock fields; compare science, not timing
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in {"latency_s", "versions"}}
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def main() -> None:
    a = run_once("results/verify_a.json")
    b = run_once("results/verify_b.json")
    keys = ["book", "hidden_patterns", "cost_latency"]
    # cost_latency synthetic uses seeded RNG -> must match; book/hidden deterministic
    ok = True
    for k in keys:
        sa = json.dumps(_strip_volatile(a[k]), sort_keys=True)
        sb = json.dumps(_strip_volatile(b[k]), sort_keys=True)
        if sa != sb:
            print(f"MISMATCH in {k}")
            ok = False
    print("REPRODUCIBLE" if ok else "NOT-REPRODUCIBLE")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
