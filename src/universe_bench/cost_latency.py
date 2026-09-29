"""Cost/latency/energy telemetry (2026 production checklist)."""
from __future__ import annotations
import time


def measure(fn, *args, **kwargs) -> dict:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return {"output": out, "latency_s": dt}


def cost_per_1k(prompt_tokens: int, gen_tokens: int, price_in_per_1k: float, price_out_per_1k: float) -> float:
    return prompt_tokens / 1000 * price_in_per_1k + gen_tokens / 1000 * price_out_per_1k


def tokens_per_sec(total_tokens: int, latency_s: float) -> float:
    return total_tokens / max(1e-9, latency_s)


def co2e_per_1m_tokens_estimate(tokens: int = 1_000_000, gpu_watts: float = 400, hours_per_1m: float = 0.05, ci_kg_per_kwh: float = 0.4) -> float:
    # Very rough teaching estimate: energy = power * time; CO2e = energy * carbon intensity
    kwh = gpu_watts * hours_per_1m / 1000
    return round(kwh * ci_kg_per_kwh, 4)


def summarize_runs(latencies: list[float]) -> dict:
    import statistics

    s = sorted(latencies)
    def pct(p: float) -> float:
        if not s:
            return 0.0
        k = (len(s) - 1) * p / 100
        f, c = int(k), min(len(s) - 1, int(k) + 1)
        return s[f] + (s[c] - s[f]) * (k - f)
    return {
        "n": len(s),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(s) if s else 0.0,
    }
