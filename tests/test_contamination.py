from universe_bench import contamination as C


def test_canary():
    assert C.has_canary("x " + C.CANARY)
    assert not C.has_canary("clean text")


def test_ngram_overlap_hit_and_miss():
    hit = C.ngram_overlap(["the quick brown fox jumps"], "the quick brown fox jumps", n=3)
    assert hit["hit"]
    miss = C.ngram_overlap(["totally different words here now"], "the quick brown fox jumps over", n=5)
    assert not miss["hit"]


def test_date_tagged():
    r = C.date_tagged_eval([{"date": "2026-01-01"}, {"date": "2024-01-01"}], cutoff="2025-01-01")
    assert r == {"total": 2, "fresh": 1, "fresh_ratio": 0.5}
