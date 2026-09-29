from universe_bench.pipeline import run_book_suite, run_hidden_patterns, run_all_and_save
from universe_bench import hallucination_lifecycle as H


def test_book_suite_deterministic(tmp_path):
    a = run_book_suite(seed=1337)
    b = run_book_suite(seed=1337)
    assert a["preds"] == b["preds"] and a["accuracy"] == b["accuracy"]


def test_hidden_patterns_shape():
    h = run_hidden_patterns()
    assert "token_bias" in h and "contamination_demo" in h and "hallucination_demo" in h
    assert H.lexical_faithfulness("a b c", "a b c")["coverage"] == 1.0


def test_run_all_saves(tmp_path):
    out = str(tmp_path / "r.json")
    rep = run_all_and_save(out, seed=1337)
    assert "versions" in rep and "book" in rep
