from universe_bench import irt_anchor as I


def test_theta_orders_ability():
    diffs = [-1.0, 0.0, 1.0] * 10
    hi = [1] * len(diffs)
    lo = [0] * len(diffs)
    assert I.estimate_theta(hi, diffs) > I.estimate_theta(lo, diffs)


def test_spearman_perfect():
    assert abs(I.spearman([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert abs(I.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-9
