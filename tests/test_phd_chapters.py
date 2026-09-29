from universe_bench import phd_chapters as P


def test_ch1_shape():
    r = P.ch1_tokenizer_confounding(1337)
    assert "acc_base" in r and "ece_base" in r and r["token_len_ratio_hello"] == 3.0


def test_ch2_sequential_irt():
    r = P.ch2_sequential_irt(1337)
    assert r["sequential"]["n_used"] <= 200
    assert set(r["irt_ablation"].keys()) == {"20", "50", "100"}
    # ranking should be largely preserved with 100 anchors
    assert r["irt_ablation"]["100"]["spearman_vs_full"] >= 0.9


def test_ch3_ch4():
    r3 = P.ch3_hallucination_rag_noise(1337)
    assert "0pct" in r3["coverage_by_noise"] and "80pct" in r3["coverage_by_noise"]
    r4 = P.ch4_perturbation_position(1337)
    assert r4["lost_in_middle_scores"]["middle"] <= r4["lost_in_middle_scores"]["start"]


def test_ch5_multilingual_gap():
    r = P.ch5_multilingual(1337)
    assert r["by_lang"]["en"]["acc"] >= r["by_lang"]["de"]["acc"] or r["by_lang"]["en"]["acc"] == 1.0


def test_ch6_kappa_range():
    r = P.ch6_judge_audit(1337)
    assert -1 <= r["cohen_kappa"] <= 1 and r["bradley_terry"]["n"] == 20


def test_ch7_gates():
    r = P.ch7_holdout_gates_tripod(1337)
    assert r["holdout_n"] == 10 and r["public_n"] == 90
    assert r["system"]["gates"]["pass_p95"] is True
