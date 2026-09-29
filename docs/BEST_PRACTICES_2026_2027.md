# 2026–2027 Best Practices (voted across 20+ sources, one-at-a-time search)

No hand-edit without verify: every claim below maps to code in `src/` + test in `tests/`.

## 1. Reproducibility is the paper
- Evalution Aug-2026 (10 engines, 153 families, 215 GPU regression tests): **benchmark-faithful defaults + versions map**. We copy: `versions.versions_map()` stored in every report.
- llm-d-benchmark: 4 phases Standup→Smoketest→Run→Teardown + DoE matrix. We copy: frozen prompts/splits/seeds + `scripts/verify_repro.py` (run twice, diff).
- NSF rubric 2026 + EduFairBench: code alone is NOT enough (gated weights, mutable APIs, hidden system prompts, backend drift). We copy: offline-first, no API required, `PYTHONHASHSEED=0 + UB_SEED`.
- Docker 2026 playbook: env parity, pin CUDA/Python, layer-cache models, deterministic seeds, non-root, `read_only /data`, `network:none` for untrusted, limits 2 CPU/8G. See `Dockerfile` + `docker-compose.yml`.

## 2. Statistics or it didn't happen
- PyData Amsterdam 2026 (Bayesian sequential): fixed-N wastes compute; use Beta-Binomial posterior `P(A>B)` + early stop at 0.95. See `sequential_bayes.py`.
- ASOasis guide + AICE-Lab: bootstrap CI95, McNemar/permutation, ECE/Brier calibration, Bradley-Terry for pairwise. See `metrics.py`.
- Growing Pains 2026 (IRT fixed anchors): 100 anchors predict full eval within 2–3pp, Spearman≥0.9. See `irt_anchor.py`.

## 3. Contamination-aware
- Sainz EMNLP23 + Balloccu EACL24 + TaskContamination AAAI24: 13-gram overlap + canary + date-tagged eval (LiveCodeBench) + private holdout (HLE). See `contamination.py` + `suite_contamination.yaml`.
- Wiki lifecycle: inception→growth→saturation→renewal. MMLU 6.5% errors / 57% virology flagged; SWE-Verified 60.8% leakage + 47.9% weak tests → report stratified slices, never single number.

## 4. System + safety
- Track p50/p95/p99, tokens/sec, $/1k, CO2e/1M (ASOasis + HELM efficiency). See `cost_latency.py`.
- OpenHalDet + lifecycle survey 2608.26168: factuality vs faithfulness (intrinsic/extrinsic), gray/black/white-box detectors, standardize prompt→generation→annotation→scoring. See `hallucination_lifecycle.py`.
- Judge validation: blinded order-randomized, hidden golds, judge-human correlation (PyInns DeepEval+RAGAS).

## 5. What to run
```bash
pip install -r requirements.txt && pip install -e .
pytest -q
python scripts/run_all.py --out results/universe_report.json
python scripts/verify_repro.py
docker compose config && docker compose up --abort-on-container-exit bench
```
