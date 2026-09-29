from universe_bench import metrics as M


def test_accuracy_and_f1():
    yt = ["pos", "pos", "neg", "neg"]
    yp = ["pos", "neg", "neg", "neg"]
    assert M.accuracy(yt, yp) == 0.75
    assert 0 <= M.macro_f1(yt, yp) <= 1


def test_pass_at_k():
    # N=10, c=2, k=1 -> 0.2 ; k=10 with c>0 and N-c<k -> 1.0
    assert abs(M.pass_at_k_unbiased(10, 2, 1) - 0.2) < 1e-9
    assert M.pass_at_k_unbiased(10, 0, 1) == 0.0
    assert M.pass_at_k_unbiased(5, 1, 5) == 1.0


def test_bootstrap_ci_contains_mean():
    vals = [1.0, 0.0, 1.0, 1.0, 0.0] * 4
    mean, lo, hi = M.bootstrap_ci(vals, n_boot=200, seed=1)
    assert lo <= mean <= hi


def test_mcnemar_no_crash():
    r = M.mcnemar(8, 2)
    assert 0 <= r["p"] <= 1


def test_ece_brier_range():
    r = M.ece_brier([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.4])
    assert 0 <= r["brier"] <= 1 and 0 <= r["ece"] <= 1
