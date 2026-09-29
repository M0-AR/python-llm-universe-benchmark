"""Book-grounded synthetic datasets (Ch.1-6) + frozen prompts. No network needed."""
from __future__ import annotations
from dataclasses import dataclass


FROZEN_PROMPTS = {
    "sentiment_v1": "Classify sentiment as positive/negative. Answer with one word. Text: {text}",
    "summarize_v1": "Summarize in <=20 words, extractive only (no new facts). Article: {text}",
    "translate_v1": "Translate EN->DE literally, keep numbers unchanged. Text: {text}",
    "qa_v1": "Answer using ONLY the context. If absent reply EXACTLY: NOT IN CONTEXT. Context: {ctx} Q: {q}",
}

# Tiny balanced set mirroring book examples (spam/sentiment/news style)
SENTIMENT_DATA = [
    ("I love this phone. It has an amazing camera!", "positive"),
    ("Beautiful and spectacular performance", "positive"),
    ("Boring and awful, act now urgent reply", "negative"),  # spam-ish negative
    ("The movie was boring and awful", "negative"),
    ("What a beautiful spectacular day", "positive"),
    ("Urgent reply needed, this is awful", "negative"),
]

QA_DATA = [
    {"ctx": "ELIZA was created in 1966 by Joseph Weizenbaum at MIT.", "q": "Who created ELIZA?", "a": "Joseph Weizenbaum"},
    {"ctx": "BERT uses only the encoder. GPT uses decoder.", "q": "Which part does BERT use?", "a": "encoder"},
    {"ctx": "Python 3.11 was released Oct 24 2022.", "q": "When was Python 3.11 released?", "a": "Oct 24 2022"},
]

TRANSLATE_PAIRS = [
    ("Hello, how is it going for you?", "Hallo, wie geht es dir?"),
    ("The quick brown fox jumps over the lazy dog.", "Der schnelle braune Fuchs springt über den faulen Hund."),
]


@dataclass(frozen=True)
class Split:
    train: list
    test: list


def sentiment_split(seed: int = 1337) -> Split:
    # Deterministic: first 4 train, last 2 test (frozen, no shuffle-on-test)
    return Split(train=SENTIMENT_DATA[:4], test=SENTIMENT_DATA[4:])
