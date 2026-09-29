# PhD Experiments Report (offline, seeded, frozen)

## ch1: ch1_tokenizer_confounding
- claim: Tokenizer bias exists (3x tokens) even when lowercasing masks downstream accuracy; version tokenizer + preprocessing or scores incomparable.
- acc base/upper/space/cs: 1.0/1.0/1.0/0.5 masking=TF-IDF lowercase=True masks case bias (delta 0.0) while BPE len_ratio=3.0 proves tokenizer bias; case-sensitive control exposes drop. Preprocessing choices hide

## ch2: ch2_sequential_irt
- claim: Close 15pp gap often needs full budget (honest undecided); clear 35pp gap early-stops (~60-80% savings). 50-100 fixed anchors recover ranking (rho>=0.9).
- sequential close: n=200 undecided at budget p=0.846 save=0.0 | clear: n=64 A better (early stop) p=0.955 save=0.68
- irt ablation: {'20': {'thetas': [-0.75, -1.0, -0.5, -0.2, 1.35], 'spearman_vs_full': 0.9, 'mae_theta_vs_acc': 0.97}, '50': {'thetas': [-0.85, -0.55, -0.2, -0.0, 1.3], 'spearman_vs_full': 1.0, 'mae_theta_vs_acc': 0.79}, '100': {'thetas': [-0.85, -0.55, -0.05, 0.25, 1.3], 'spearman_vs_full': 1.0, 'mae_theta_vs_acc': 0.71}}

## ch3: ch3_hallucination_rag
- claim: Grounding lifts coverage; 80% noise collapses noisy-claim discrimination -> need retrieval precision + human 5% audit.
- coverage_by_noise keys: ['0pct', '20pct', '80pct'] rag_lift={'rag_off_coverage': 0.0, 'rag_on_coverage': 1.0}

## ch4: ch4_perturbation_position
- claim: Middle position degrades retrieval; 80% noise collapses accuracy (Arabic-RAG lesson).
- lost_in_middle={'start': 1.0, 'middle': 0.8, 'end': 1.0} noise_acc={'0pct': 1.0, '20pct': 0.85, '80pct': 0.45}

## ch5: ch5_multilingual
- claim: EN-trained TF-IDF collapses on DE/UR-roman -> multilingual eval required (Frontier/Urdu lesson).
- by_lang acc: en={'acc': 1.0, 'preds': ['positive', 'negative'], 'gold': ['positive', 'negative']} de={'acc': 0.5, 'preds': ['positive', 'positive'], 'gold': ['positive', 'negative']} ur={'acc': 1.0, 'preds': ['positive', 'negative'], 'gold': ['positive', 'negative']} gap=0.5

## ch6: ch6_judge_audit
- claim: Report kappa + CI + BT/Elo; randomize order + hidden golds (DeepEval/RAGAS lesson).
- kappa=0.7 win_rate={'mean': 0.45, 'ci95': [0.25, 0.65]} bt={'win_rate_a': 0.45, 'elo_diff_a_minus_b': -34.9, 'n': 20}

## ch7: ch7_holdout_gates
- claim: Private 10% + date-tag + p95/cost/CO2e gates in CI prevent leakage + cost blowout.
- holdout=10 public=90 fresh={'total': 90, 'fresh': 57, 'fresh_ratio': 0.6333333333333333} gates={'p95_s_max': 0.5, 'cost_per_1k_max': 0.05, 'co2e_per_1m_max': 0.05, 'pass_p95': True, 'pass_cost': True, 'pass_co2e': True}

versions: 3.12.13 (main, May  8 2026, 18:32:55) [Clang 22.1.3 ]