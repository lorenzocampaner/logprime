"""
Text parser: raw text → clean sentence list ready for LogPrime.

Pipeline:
  1. Split into sentences (on . ! ? ; and blank lines; single line breaks
     inside a paragraph are treated as spaces, so hard-wrapped text works)
  2. Lowercase
  3. Expand apostrophes: keep only the part after (l'uomo → uomo, dell'aria → aria)
  4. Remove remaining punctuation and digits (configurable)
  5. Collapse whitespace
  6. Drop sentences shorter than min_words
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


# Splits after sentence-ending punctuation, or on blank lines
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n\s*\n")

# Apostrophe: keep only the part after (handles l', dell', all', un', s', c', ...)
_APOSTROPHE = re.compile(r"\w+['\u2019](\w+)")

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_DIGITS = re.compile(r"\d+")


def parse_text(
    text: str,
    min_words: int = 2,
    remove_digits: bool = True,
    stopwords: Optional[set] = None,
) -> List[str]:
    """Parse a raw string into a list of clean sentences.

    Parameters
    ----------
    text         : raw input text.
    min_words    : discard sentences with fewer tokens than this.
    remove_digits: if True, strip digit-only tokens.
    stopwords    : optional set of words to exclude from each sentence.

    Returns
    -------
    List of cleaned sentence strings.
    """
    sentences = []
    for raw in _SENT_SPLIT.split(text):
        raw = raw.strip()
        if not raw:
            continue

        cleaned = " ".join(raw.split()).lower()
        # Expand apostrophes before removing punctuation
        cleaned = _APOSTROPHE.sub(r"\1", cleaned)
        cleaned = _PUNCT.sub(" ", cleaned)
        if remove_digits:
            cleaned = _DIGITS.sub(" ", cleaned)

        tokens = cleaned.split()
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]

        if len(tokens) >= min_words:
            sentences.append(" ".join(tokens))

    return sentences


def parse_file(
    path: str | Path,
    encoding: str = "utf-8",
    **kwargs,
) -> List[str]:
    """Parse a .txt file into clean sentences. kwargs passed to parse_text."""
    text = Path(path).read_text(encoding=encoding)
    return parse_text(text, **kwargs)


def parse_dir(
    directory: str | Path,
    encoding: str = "utf-8",
    **kwargs,
) -> List[str]:
    """Parse all .txt files in a directory. kwargs passed to parse_text."""
    files = sorted(Path(directory).glob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found in: {directory}")
    sentences = []
    for f in files:
        sentences.extend(parse_file(f, encoding=encoding, **kwargs))
    return sentences


# ------------------------------------------------------------------
# Built-in stopword lists
# ------------------------------------------------------------------

STOPWORDS_IT = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "o", "ma", "se", "che", "non", "è", "sono", "ha",
    "ho", "hai", "hanno", "del", "della", "dello", "dei", "degli",
    "delle", "al", "alla", "allo", "ai", "agli", "alle",
    "nel", "nella", "nello", "nei", "negli", "nelle",
    "sul", "sulla", "sullo", "sui", "sugli", "sulle",
    "ci", "si", "mi", "ti", "vi", "li", "ne", "anche",
    "più", "già", "come", "quando", "dove", "perché", "questo",
    "questa", "questi", "queste", "quello", "quella",
}

STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on",
    "at", "to", "for", "of", "with", "by", "from", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "not", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they",
    "my", "your", "his", "her", "our", "their", "as", "so",
    "than", "then", "there", "here", "which", "who", "what",
    "when", "where", "how", "all", "more", "also", "no", "into",
}
