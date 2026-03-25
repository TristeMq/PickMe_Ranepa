import os
import re
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parents[2] / "data"


@lru_cache(maxsize=1)
def _load_stop_words() -> frozenset[str]:
    words: set[str] = set()
    for fname in ("ru_abusive_words.txt", "ru_curse_words.txt"):
        path = DATA_DIR / fname
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
    return frozenset(words)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[а-яёa-z]+", text.lower())


def contains_profanity(text: str) -> bool:
    stop_words = _load_stop_words()
    tokens = _tokenize(text)
    return any(token in stop_words for token in tokens)
