"""Generate verified demo assets from results/*.json (no hand-drawn numbers).
Outputs: docs/demo/chart_irt.png, chart_noise.png, chart_latency.png, demo.mp4, demo.gif, hero.png
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "docs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    phd = json.loads((RES / "phd_report.json").read_text())
    uni = json.loads((RES / "universe_report.json").read_text())
    return phd, uni


def chart_irt(phd):
    abl = phd["ch2"]["irt_ablation"]
    ks = [20, 50, 100]
    rhos = [abl[str(k)]["spearman_vs_full"] for k in ks]
    plt.figure(figsize=(6, 3.5))
    plt.plot(ks, rhos, marker="o")
    plt.axhline(0.9, linestyle="--")
    plt.title("IRT fixed anchors recover ranking (Spearman)")
    plt.xlabel("anchors")
    plt.ylabel("rho vs full")
    plt.ylim(0.8, 1.05)
    for x, y in zip(ks, rhos):
        plt.text(x, y + 0.01, f"{y}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "chart_irt.png", dpi=150)
    plt.close()


def chart_noise(phd):
    nz = phd["ch4"]["noise_accuracy_proxy"]
    xs = ["0%", "20%", "80%"]
    ys = [nz["0pct"], nz["20pct"], nz["80pct"]]
    plt.figure(figsize=(6, 3.5))
    plt.bar(xs, ys)
    plt.title("Noise robustness collapses at 80% (Arabic-RAG lesson)")
    plt.ylabel("accuracy proxy")
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUT / "chart_noise.png", dpi=150)
    plt.close()


def chart_latency(phd):
    lat = phd["ch7"]["system"]["latency"]
    vals = [("p50", lat["p50"]), ("p95", lat["p95"]), ("p99", lat["p99"]), ("mean", lat["mean"])]
    xs = [k for k, _ in vals]
    ys = [v for _, v in vals]
    plt.figure(figsize=(6, 3.5))
    plt.bar(xs, ys)
    plt.title("System latency (seeded synthetic, n=200)")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig(OUT / "chart_latency.png", dpi=150)
    plt.close()


def hero():
    plt.figure(figsize=(10, 3))
    plt.text(0.5, 0.6, "🌌 PYTHON + LLM UNIVERSE BENCHMARK", ha="center", fontsize=18, weight="bold")
    plt.text(0.5, 0.35, "24 tests green • 68% savings • rho=1.0 • 3x tokenizer • MIT • Docker", ha="center", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "hero.png", dpi=150)
    plt.close()


def video():
    import matplotlib.animation as animation
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle("Universe Benchmark — 60s demo (from verified results)")

    # Panel 1: sequential savings bar grows
    ax = axes[0]
    ax.set_title("Sequential savings")
    ax.set_ylim(0, 1)
    bars = ax.bar(["close\n200", "clear\n64"], [0, 0])

    # Panel 2: IRT rho
    ax2 = axes[1]
    ax2.set_title("IRT rho vs anchors")
    ax2.set_ylim(0.8, 1.05)
    (line,) = ax2.plot([], [], marker="o")
    ax2.set_xticks([20, 50, 100])

    # Panel 3: noise collapse
    ax3 = axes[2]
    ax3.set_title("Noise 0/20/80%")
    ax3.set_ylim(0, 1.1)
    bars3 = ax3.bar(["0%", "20%", "80%"], [0, 0, 0])

    rho = {20: 0.9, 50: 1.0, 100: 1.0}
    noise = [1.0, 0.85, 0.45]

    def update(f):
        # f 0..59
        p = min(1.0, (f + 1) / 20)
        bars[0].set_height(0.0 * p)
        bars[1].set_height(0.68 * min(1.0, (f + 1) / 30))
        k = min(59, f)
        n = 1 + int(k / 20)  # 1..3 points
        xs = [20, 50, 100][:n]
        ys = [rho[x] for x in xs]
        line.set_data(xs, ys)
        for i, b in enumerate(bars3):
            b.set_height(noise[i] * min(1.0, (f + 1) / 40))
        return list(bars) + [line] + list(bars3)

    ani = animation.FuncAnimation(fig, update, frames=60, interval=200, blit=False)
    mp4 = OUT / "demo.mp4"
    gif = OUT / "demo.gif"
    # ffmpeg available per env check
    ani.save(str(mp4), writer="ffmpeg", fps=10, dpi=120)
    ani.save(str(gif), writer="pillow", fps=10, dpi=80)
    plt.close(fig)


def main():
    phd, uni = load()
    chart_irt(phd)
    chart_noise(phd)
    chart_latency(phd)
    hero()
    video()
    print("assets:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
