"""
Semantic: dense word vectors derived from a LogPrime map.

The map's sums already hold, for every pair of words, how many times one
follows the other in a sentence (S[i][j] / p_j). Adding the transpose gives
sentence-level co-occurrence counts, without reading the text again. From
those: positive PMI, with the context distribution smoothed by α = 0.75, then a
truncated SVD. This is the classic count-based model (Levy, Goldberg & Dagan,
2015) on the counts LogPrime keeps anyway.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

if TYPE_CHECKING:
    from .core import LogPrime

PathLike = Union[str, Path]
_FORMAT_VERSION = 1


def cooccurrence(lp: "LogPrime") -> sp.csr_matrix:
    """Symmetric sentence co-occurrence counts read off the map, diagonal excluded."""
    S = lp._sums.tocoo()
    counts = S.data / np.asarray(lp._primes, dtype=np.float64)[S.col]
    off = S.row != S.col
    C = sp.csr_matrix((counts[off], (S.row[off], S.col[off])), shape=S.shape)
    return (C + C.T).tocsr()


def ppmi(C: sp.spmatrix, alpha: float = 0.75) -> sp.csr_matrix:
    """Positive pointwise mutual information, with context smoothing α."""
    C = C.tocoo()
    total = C.sum()
    w = np.asarray(C.sum(axis=1)).ravel()
    c = np.asarray(C.sum(axis=0)).ravel() ** alpha
    c = c / c.sum()
    pmi = np.log(C.data / total) - np.log(w[C.row] / total) - np.log(c[C.col])
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)


class Semantic:
    """Unit-length word vectors, one row per word.

    counts : how often each word was seen (from the map's self cells; a word
             repeated inside a sentence counts a little more). Used to keep
             rare, noisy words out of `neighbours`.
    """

    def __init__(self, words: Sequence[str], vectors: np.ndarray, counts: Sequence[int],
                 normalize: bool = True) -> None:
        self.words: List[str] = list(words)
        self.index: Dict[str, int] = {w: i for i, w in enumerate(self.words)}
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.int64)

    @classmethod
    def from_map(cls, lp: "LogPrime", dim: int = 300, alpha: float = 0.75,
                 power: float = 0.5, seed: int = 0) -> "Semantic":
        """PPMI + SVD on the map's co-occurrence counts.

        dim   : vector size (at most n_words - 1).
        power : singular values are raised to this power (0.5 is the usual choice).
        """
        P = ppmi(cooccurrence(lp), alpha)
        dim = min(dim, min(P.shape) - 1)
        v0 = np.random.default_rng(seed).standard_normal(min(P.shape))
        U, s, _ = svds(P, k=dim, v0=v0)
        counts = lp._sums.diagonal() // np.asarray(lp._primes, dtype=np.int64)
        return cls(lp.words, U * s**power, counts)

    def vector(self, text: str) -> np.ndarray:
        """Mean of the vectors of the known words in `text` (zeros if none)."""
        ids = [self.index[t] for t in text.lower().split() if t in self.index]
        if not ids:
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        return self.vectors[ids].mean(axis=0)

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity of two words or texts (0.0 if either has no known word)."""
        va, vb = self.vector(a), self.vector(b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))

    def neighbours(self, text: str, n: int = 10, min_count: int = 5) -> List[tuple]:
        """The n words most similar to `text`, among words seen at least min_count times."""
        v = self.vector(text)
        if not v.any():
            return []
        sims = self.vectors @ (v / np.linalg.norm(v))
        sims[self.counts < min_count] = -np.inf
        for t in text.lower().split():
            if t in self.index:
                sims[self.index[t]] = -np.inf
        best = np.argsort(-sims)[:n]
        return [(self.words[i], float(sims[i])) for i in best if np.isfinite(sims[i])]

    def save(self, path: PathLike) -> None:
        """Write to a .npz file."""
        np.savez_compressed(path, vectors=self.vectors, counts=self.counts,
                            meta=json.dumps({"format": _FORMAT_VERSION, "words": self.words}))

    @classmethod
    def load(cls, path: PathLike) -> "Semantic":
        with np.load(path) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("format") != _FORMAT_VERSION:
                raise ValueError(f"Unsupported semantic format in '{path}'.")
            return cls(meta["words"], f["vectors"], f["counts"], normalize=False)

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        return f"Semantic({len(self)} words, dim={self.vectors.shape[1]})"
