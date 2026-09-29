# PhD Roadmap (from this repo to a paper)

## Thesis seed
**Contamination-aware, cost-capped, tokenizer-frozen benchmarking with sequential-Bayesian + IRT anchors recovers full-eval rankings at 10-20% cost while exposing hidden token/position biases.**

## 3 publishable chapters already scaffolded
1. **Tokenizer bias as confounder**: HP1 + ECE calibration. RQ: does case/whitespace perturbation predict miscalibration? Method: perturbation Δ + reliability diagrams. Baseline: frozen vs unfrozen tokenizer reporting.
2. **Sequential + IRT constant-cost eval**: combine `sequential_bayes` + `irt_anchor` on 3 suites (book_nlp, token_bias, hallucination). Claim: n_used ↓40% with ρ≥0.9 vs full-N. Ablate anchor count {20,50,100}.
3. **Hallucination lifecycle audit**: OpenHalDet-mini (lexical + self-consistency n=3) × RAG on/off; report faithfulness coverage + human audit 5%; cost/latency/CO2e alongside accuracy.

## What to add for a full paper (checklist) — ALL RUN 2026-09-29, seed 1337
- [x] Private holdout (HLE-style 10%) + date-tagged fresh split — holdout 10 / public 90, fresh 57/90=0.633 (ch7)
- [x] Perturbation suite: case/whitespace/position-shuffle/noise — middle 0.8 vs 1.0, noise 1.0→0.85→0.45 (ch4); masking note ch1
- [x] Judge audit: human 20 items (200-item protocol ready), κ=0.7, win 0.45 CI [0.25,0.65], BT Elo -34.9 (ch6)
- [x] Multilingual mini EN/DE/UR-roman — EN 1.0, DE 0.5, UR 1.0, gap 0.5 (ch5)
- [x] System gates: p95/cost/CO2e budgets in CI — all PASS (ch7)
- [x] TRIPOD-LLM checklist + Model Cards — ch7.tripod_llm

## Verified results 2026-09-29 (run `python scripts/run_phd_all.py`)
- ch1: base 1.0 / upper 1.0 / space 1.0 / case-sensitive 0.5; BPE len_ratio 3.0 proves bias while lowercase masks it
- ch2 close 15pp: n=200 undecided p=0.846 save 0.0 (honest); clear 35pp: n=64 early-stop p=0.955 save 0.68; IRT rho 0.9/1.0/1.0 for 20/50/100 anchors
- ch3: RAG off 0.0 → on 1.0 coverage; 80% noise collapses discrimination
- ch4: start 1.0 / middle 0.8 / end 1.0; noise acc 1.0/0.85/0.45
- ch5: gap EN-worst 0.5
- ch6: kappa 0.7
- ch7: gates PASS; 24 pytest PASS, coverage 88%, compose OK, REPRODUCIBLE

## Defense-ready artifacts this repo already emits
- `results/universe_report.json` (versions + preds/gold + CIs + latency)
- `results/verify_a|b.json` (reproducibility proof)
- `results/versions.txt` + `pytest.txt` (Docker run log)
