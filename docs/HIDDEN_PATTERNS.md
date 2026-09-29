# Hidden Patterns (teachable, verifiable, PhD-seed)

## HP1 — Tokenizer is the model (PROVEN in code)
- `hello`(1) vs `HELLO`(3: HE/LL/O), `200`(1) vs `201`(2: 20/1), `" last"` merged vs `" "+"last"` split.
- Verify: `pytest tests/test_tokenization_bias.py -q`
- Why it matters: adjacent integers / casing change token count → arithmetic + multilingual brittleness (Yiddish noisy-web failure, Urdu gaps). Always freeze + report tokenizer + template.
- PhD angle: measure ΔP(next-token) under whitespace/case perturbations; correlate with ECE.

## HP2 — Contamination hides as accuracy
- 13-gram overlap + canary + date-tagged fresh ratio. Demo hits 1.0 on copied ELIZA sentence, 0.0 on novel.
- Verify: `pytest tests/test_contamination.py -q`
- PhD angle: contamination-adjusted score = raw − λ·overlap; study λ stability across suites.

## HP3 — Fixed-N wastes 30-60% budget
- Beta-Binomial `P(A>B)` early-stops when >0.95/<0.05. Extreme 18/20 vs 2/20 → P>0.95.
- Verify: `pytest tests/test_sequential_bayes.py -q`
- PhD angle: sequential + IRT anchors = constant cost per new dataset (Growing Pains) with calibrated stopping.

## HP4 — 100 anchors recover ranking (ρ≥0.9)
- Rasch `sigmoid(theta-b)` MLE; perfect Spearman on monotonic demo.
- Verify: `pytest tests/test_irt_anchor.py -q`
- PhD angle: anchor selection (uncertainty vs diversity) under distribution shift.

## HP5 — Hallucinations peak at semantic-phase onsets
- PSRD 2026: 50% reduction on LLaVA-1.5-7B via phase-wise self-reward. Our proxy: lexical coverage faithful≥0.6 / extrinsic 0.3-0.6 / hallucinated<0.3.
- Verify: run `python scripts/run_all.py`, inspect `hidden_patterns.hallucination_demo`.
- PhD angle: lifecycle taxonomy (data/training/inference) × factuality/faithfulness matrix; human audit ≥5%.

## HP6 — Position + verbosity bias (roadmap)
- Lost-in-middle + Arena length/markdown bias. Next: add position-shuffle + length-normalized win-rate (Bradley-Terry) experiments.
