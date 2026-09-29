from universe_bench import sequential_bayes as S


def test_prob_a_better_extreme():
    # A 18/20 vs B 2/20 -> P(A>B) should be very high
    p = S.prob_a_better(18, 20, 2, 20, draws=3000, seed=1)
    assert p > 0.95


def test_sequential_early_stop():
    a = [1] * 30 + [1] * 10
    b = [0] * 30 + [0] * 10
    r = S.sequential_decision(a, b, thresh=0.95, min_n=10, max_n=40)
    assert r["n_used"] <= 40 and "decision" in r
