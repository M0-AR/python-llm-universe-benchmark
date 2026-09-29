from universe_bench import tokenization_bias as T


def test_case_bias():
    d = T.case_bias_demo()
    assert d["hello"] == ["hello"] and d["HELLO"] == ["HE", "LL", "O"]
    assert d["len_ratio"] == 3.0


def test_numeric_bias():
    d = T.numeric_bias_demo()
    assert d["200"] == ["200"] and d["201"] == ["20", "1"]


def test_whitespace():
    d = T.whitespace_demo()
    assert d["merged"] == [" last"] and d["split"] == [" ", "last"]
