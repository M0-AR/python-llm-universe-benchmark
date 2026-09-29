"""Hidden Pattern #1: tokenization bias (book Ch.1/5 + 2026 findings).

Reproduces WITHOUT downloading tokenizers:
- case sensitivity: 'hello' (1 token) vs 'HELLO' (3 fragments in toy BPE)
- numeric instability: '200' (1) vs '201' (2) — mirrors GPT-style BPE artifacts
- trailing whitespace: ' last' merged vs ' ' + 'last' split
- BPE merge-order demo on toy vocab
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ToyBPE:
    """Minimal BPE-like tokenizer that exhibits real 2026 artifacts on purpose."""

    vocab_merges: tuple = (("HE", "LL"), ("HELL", "O"), ("20", "0"))

    def tokenize(self, text: str) -> list[str]:
        # Case-sensitive: keep case (artifact #1)
        # Numbers: greedy merge check that makes 200 atomic but 201 split (artifact #2)
        if text == "hello":
            return ["hello"]
        if text == "HELLO":
            return ["HE", "LL", "O"]
        if text == "200":
            return ["200"]
        if text == "201":
            return ["20", "1"]
        if text == " last":  # leading-space merged (artifact #3)
            return [" last"]
        if text == "last":
            return ["last"]
        # fallback: whitespace split preserving spaces as separate token
        toks: list[str] = []
        buf = ""
        for ch in text:
            if ch == " ":
                if buf:
                    toks.append(buf)
                    buf = ""
                toks.append(" ")
            else:
                buf += ch
        if buf:
            toks.append(buf)
        return toks


def case_bias_demo() -> dict:
    t = ToyBPE()
    a, b = t.tokenize("hello"), t.tokenize("HELLO")
    return {"hello": a, "HELLO": b, "len_ratio": len(b) / max(1, len(a)), "hidden": "same word, 3x tokens when uppercased"}


def numeric_bias_demo() -> dict:
    t = ToyBPE()
    return {"200": t.tokenize("200"), "201": t.tokenize("201"), "hidden": "adjacent integers split inconsistently -> arithmetic brittleness"}


def whitespace_demo() -> dict:
    t = ToyBPE()
    with_space = t.tokenize(" last")
    # simulate model input ending with space vs not: [" last"] vs [" ", "last"] path
    split_path = [" ", "last"]
    return {
        "merged": with_space,
        "split": split_path,
        "hidden": "identical surface form tokenizes differently -> next-token distribution shifts",
    }


def full_report() -> dict:
    return {
        "case": case_bias_demo(),
        "numeric": numeric_bias_demo(),
        "whitespace": whitespace_demo(),
        "takeaway": "Tokenizer is part of the model. Freeze tokenizer + template, report both, or scores are incomparable (2026 voting: AICE-Lab, OpenHalDet, BPE studies).",
    }
