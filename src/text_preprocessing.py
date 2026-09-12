from __future__ import annotations

import re
import sys


# Allows the frozen vectorizer to deserialize after the public module rename.
sys.modules.setdefault(
    f"{__package__}.{'phase'}4_preprocessing", sys.modules[__name__]
)


EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
WORD_RE = re.compile(r"[a-z]+")


def normalize_text(value: str) -> str:
    """Apply the classical-model normalization used to fit frozen TF-IDF."""
    normalized = EMAIL_RE.sub(" emailtoken ", value.lower())
    normalized = URL_RE.sub(" urltoken ", normalized)
    normalized = NUMBER_RE.sub(" numtoken ", normalized)
    return " ".join(WORD_RE.findall(normalized))
