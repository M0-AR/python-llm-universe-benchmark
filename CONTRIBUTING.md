# Contributing

1. Fork → `git checkout -b feature/your-feature`
2. `pip install -r requirements.txt && pip install -e .`
3. `PYTHONHASHSEED=0 pytest -q`
4. `python scripts/run_all.py --out results/universe_report.json && python scripts/run_phd_all.py --out results/phd_report.json --md results/phd_report.md && python scripts/verify_repro.py`
5. If you touch README numbers, re-run `python scripts/make_demo_assets.py` (charts/video come from JSON, never hand-drawn).
6. Submit PR with what changed + test log. We respond within 48h.

Rule: **no hand-edit without verify** — read back every edited file, keep `docker compose config` green.
