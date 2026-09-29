# Voting Matrix (different opinions, one search at a time — no parallel calls)

| Source (2026) | Vote | What we adopted |
|---|---|---|
| Evalution v0.0.14 | 153 families, benchmark-faithful, versions map, DockerAgentRuntime | versions map + frozen prompts + sandbox compose |
| llm-d-benchmark | Step abstraction, 4 phases, DoE | verify_repro + frozen splits |
| PyData Bayesian sequential | Fixed-N wasteful, BF stopping | sequential_bayes.py |
| lm-eval-harness (Eleuther/Nous) | 60+ tasks, vLLM/API, log_samples | task YAMLs + log preds/gold |
| tainguyen07 harness | prompt registry, LLM-judge, BT scores, 91% cov, gates | FROZEN_PROMPTS + judge audit note + CI |
| AlephAlpha eval-framework | BaseLLM/Task/Metric, perturbations, Docker | suite YAMLs + perturbation idea in roadmap |
| Growing Pains IRT | 100 anchors → ±2-3pp, ρ≥0.9, 400 models | irt_anchor.py |
| PyInns DeepEval+RAGAS | Polars, cost/token, latency dashboard | cost_latency.py |
| Docker guides (PremAI/AltQA/Collabnix) | parity, pin CUDA, non-root, 8G/2CPU, read-only | Dockerfile/compose |
| EduFairBench/NSF | stability/uncertainty/error patterns, code≠repro | offline + seeds + CIs |
| AICE-Lab complete guide | 4 failures: saturation/contamination/single-number/harness-dependence | portfolio of suites, never 1 number |
| Wiki benchmark | pass@k unbiased 1-C(N-c,k)/C(N,k), lifecycle, Goodhart | metrics.pass_at_k_unbiased |
| OpenHalDet / lifecycle survey | standardize pipeline, black/gray/white detectors | hallucination_lifecycle.py |
| Token-bias EMNLP24 + BPE studies | case/numeric/whitespace artifacts | tokenization_bias.py |
| Lost-in-middle + phase-wise hallucination | position/phase effects | roadmap experiments |
| Kaggle Urdu/Frontier/AI-vs-Human | multilingual + speed + detection datasets | roadmap + references |

Disagreements kept: single leaderboard vs portfolio (we side portfolio); fixed-N vs sequential (we side sequential with budget cap); LLM-judge vs human-only (we side hybrid with correlation audit).
