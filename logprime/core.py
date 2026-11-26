"""
LogPrime: prime-number-based word embeddings.

Every word gets a distinct prime, in order of first appearance. For each
occurrence of a word w_i (prime p_i) in a sentence, every word w_j that
follows it adds its prime p_j to the cell (w_i, w_j); the word itself adds
p_i to its own cell (w_i, w_i). After the whole corpus:

    S[w_i][w_j] = Σ p_j                       (exact integer sum)
    M[w_i][w_j] = ln(p_i · S[w_i][w_j])       (the embedding value)

The row M[w_i] is the embedding of w_i. A sentence vector is the sum of the
rows of its words, and Orthogonal Matching Pursuit (OMP) recovers the words
from that vector.

Only S is stored. Because it is a plain sum, adding new text to a map gives
exactly the same result as rebuilding it from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import scipy.sparse as sp

from .primes import primes_after

if TYPE_CHECKING:
    from .codec import Codec
    from .semantic import Semantic

PathLike = Union[str, Path]

_FORMAT_VERSION = 1
_CHUNK = 20_000  # sentences accumulated before folding into the sparse matrix


_TRIU: Dict[int, tuple] = {}


def _forward_pairs(ids: np.ndarray, lengths: np.ndarray) -> tuple:
    """(row, col) word-id pairs for every word and each word at or after it.

    `ids` holds the word ids of several sentences back to back; `lengths`
    gives each sentence's length.
    """
    rows, cols = [], []
    start = 0
    for L in lengths.tolist():
        if L not in _TRIU:
            _TRIU[L] = np.triu_indices(L)
        a, b = _TRIU[L]
        sent = ids[start : start + L]
        rows.append(sent[a])
        cols.append(sent[b])
        start += L
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


class LogPrime:
    """A corpus map: vocabulary of word → prime, plus the co-occurrence sums."""

    def __init__(self) -> None:
        self.vocabulary: Dict[str, int] = {}
        self._words: List[str] = []
        self._primes: List[int] = []
        self._position: Dict[str, int] = {}
        self._next_prime = primes_after(1)
        self._sums = sp.csr_matrix((0, 0), dtype=np.int64)
        self._matrix: Optional[sp.csr_matrix] = None
        self._row_norms: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _index_of(self, word: str) -> int:
        """Row index of `word`, assigning it the next prime if it is new."""
        idx = self._position.get(word)
        if idx is None:
            idx = len(self._words)
            p = next(self._next_prime)
            self.vocabulary[word] = p
            self._words.append(word)
            self._primes.append(p)
            self._position[word] = idx
        return idx

    def add(self, sentences: Iterable[str]) -> "LogPrime":
        """Add sentences (whitespace-separated words) to the map.

        Can be called any number of times; the result is the same as a single
        call with all the sentences.
        """
        chunk: List[np.ndarray] = []

        def flush() -> None:
            n = len(self._words)
            ids = np.concatenate(chunk) if chunk else np.zeros(0, dtype=np.int64)
            lengths = np.array([len(c) for c in chunk], dtype=np.int64)
            chunk.clear()
            rows, cols = _forward_pairs(ids, lengths)
            primes = np.asarray(self._primes, dtype=np.int64)
            new = sp.csr_matrix((primes[cols], (rows, cols)), shape=(n, n), dtype=np.int64)
            old = self._sums
            old.resize((n, n))
            self._sums = (old + new).tocsr()

        for k, sentence in enumerate(sentences, 1):
            chunk.append(np.array([self._index_of(t) for t in sentence.lower().split()], dtype=np.int64))
            if k % _CHUNK == 0:
                flush()
        flush()

        self._matrix = None
        self._row_norms = None
        return self

    # ------------------------------------------------------------------
    # Embedding matrix
    # ------------------------------------------------------------------

    @property
    def words(self) -> List[str]:
        """Vocabulary in prime order: row and column labels of the matrix."""
        return self._words

    @property
    def matrix(self) -> sp.csr_matrix:
        """Sparse embedding matrix M, shape (n_words, n_words)."""
        if self._matrix is None:
            S = self._sums.tocsr()
            S.sort_indices()
            row_of = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
            p_row = np.asarray(self._primes, dtype=np.float64)[row_of]
            data = np.log(p_row) + np.log(S.data.astype(np.float64))
            self._matrix = sp.csr_matrix((data, S.indices, S.indptr), shape=S.shape)
            self._row_norms = np.sqrt(np.asarray(self._matrix.power(2).sum(axis=1)).ravel())
            self._row_nnz = np.diff(S.indptr)
            self._pattern = sp.csr_matrix(
                (np.ones_like(data), S.indices, S.indptr), shape=S.shape
            )
            self._matrix_csc = self._matrix.tocsc()
            self._col_nnz = np.diff(self._matrix_csc.indptr)
            # Cache for decode: keep ~256 MB of Gram columns.
            self._gram_cache = {}
            self._gram_cache_size = max(16, (256 << 20) // (8 * max(S.shape[0], 1)))
        return self._matrix

    def embedding(self, word: str) -> Dict[str, float]:
        """The row of one word as {context_word: value}."""
        i = self._position[word.lower()]
        row = self.matrix.getrow(i)
        return {self._words[j]: float(v) for j, v in zip(row.indices, row.data)}

    def vector(self, sentence: str) -> np.ndarray:
        """Sentence vector: sum of the rows of its words. Unknown words are ignored."""
        M = self.matrix
        vec = np.zeros(M.shape[1])
        for t in sentence.lower().split():
            i = self._position.get(t)
            if i is not None:
                start, end = M.indptr[i], M.indptr[i + 1]
                vec[M.indices[start:end]] += M.data[start:end]
        return vec

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two sentences (or two words).

        Both sentence vectors are projected onto the columns of the words that
        appear in `a` or `b`, then compared. For two single words this compares
        [M[a][a], M[a][b]] with [M[b][a], M[b][b]], so it measures how strongly
        the two words co-occur relative to themselves. Returns 0.0 if either
        side has no known words.
        """
        M = self.matrix
        ids_a = [self._position[t] for t in a.lower().split() if t in self._position]
        ids_b = [self._position[t] for t in b.lower().split() if t in self._position]
        cols = list(dict.fromkeys(ids_a + ids_b))
        if not ids_a or not ids_b:
            return 0.0
        va = np.asarray(M[ids_a][:, cols].sum(axis=0)).ravel()
        vb = np.asarray(M[ids_b][:, cols].sum(axis=0)).ravel()
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, vec: np.ndarray, n_words: Optional[int] = None) -> List[str]:
        """Recover the words whose rows sum to `vec`, as a multiset (order is lost).

        Orthogonal Matching Pursuit with over-selection: words are picked
        greedily, possibly more than the sentence has. A least-squares fit then
        gives each picked word a coefficient that is close to its count
        (1, 2, …) and close to 0 for wrong picks. Rounding gives the counts.

        n_words : expected number of words, if known. It speeds up the search
                  (at most 3 × n_words picks at first); it is not enforced.

        If that first attempt does not reproduce `vec`, the search is repeated
        without the bound. Use `verify(words, vec)` to check the result: it is
        True only if the words reproduce `vec` exactly.
        """
        if n_words:
            words = self._omp(vec, 3 * n_words)
            if self.verify(words, vec):
                return words
        return self._omp(vec, None)

    def _omp(self, vec: np.ndarray, max_steps: Optional[int]) -> List[str]:
        M = self.matrix
        vec = np.asarray(vec, dtype=np.float64)
        n = M.shape[0]
        vv = float(vec @ vec)
        if n == 0 or vv == 0:
            return []

        Mv = M @ vec  # Mv[i] = <row_i, vec>
        scores_mask = np.ones(n, dtype=bool)
        if vec.min() >= 0:
            # Every value in M is positive, so with a non-negative combination
            # a word can only be present if its whole row lies inside the
            # support of vec.
            inside = self._pattern @ (vec > 0).astype(np.float64)
            scores_mask = inside == self._row_nnz

        if max_steps is None:
            max_steps = min(int(scores_mask.sum()), 1000)
        Q = np.empty((n, max_steps))  # Q[:, k] = M @ row(chosen[k])
        chosen: List[int] = []
        coeffs = np.zeros(0)
        for k in range(max_steps):
            corr = Mv - Q[:, :k] @ coeffs if k else Mv.copy()
            scores = np.where(scores_mask, corr / self._row_norms, -np.inf)
            scores[chosen] = -np.inf
            best = int(np.argmax(scores))
            if scores[best] <= 0:
                break
            chosen.append(best)
            Q[:, k] = self._gram_column(best)
            G = Q[chosen, : k + 1]
            try:
                coeffs = np.linalg.solve(G, Mv[chosen])
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(G, Mv[chosen], rcond=None)[0]
            # ‖residual‖² = ‖vec‖² − <coeffs, Mv[chosen]>, since G coeffs = Mv[chosen]
            if vv - coeffs @ Mv[chosen] <= 1e-12 * vv:
                break

        counts = np.rint(coeffs).astype(int)
        return [self._words[i] for i, c in zip(chosen, counts) for _ in range(max(c, 0))]

    def verify(self, words: Sequence[str], vec: np.ndarray, rtol: float = 1e-9) -> bool:
        """True if the words' rows sum exactly (up to float rounding) to `vec`."""
        vec = np.asarray(vec, dtype=np.float64)
        diff = vec - self.vector(" ".join(words))
        return bool(np.linalg.norm(diff) <= rtol * max(np.linalg.norm(vec), 1.0))

    def _gram_column(self, j: int) -> np.ndarray:
        """M @ row_j, i.e. the inner product of every row with row j (cached)."""
        col = self._gram_cache.get(j)
        if col is None:
            M = self.matrix
            s, e = M.indptr[j], M.indptr[j + 1]
            cols = M.indices[s:e]
            if self._col_nnz[cols].sum() < M.nnz // 4:
                # rare word: only the few columns it touches
                col = self._matrix_csc[:, cols] @ M.data[s:e]
            else:
                # frequent word: one pass over the whole matrix is cheaper
                dense = np.zeros(M.shape[1])
                dense[cols] = M.data[s:e]
                col = M @ dense
            if len(self._gram_cache) >= self._gram_cache_size:
                self._gram_cache.pop(next(iter(self._gram_cache)))
            self._gram_cache[j] = col
        else:
            self._gram_cache[j] = self._gram_cache.pop(j)  # mark as recently used
        return col

    # ------------------------------------------------------------------
    # Derived models
    # ------------------------------------------------------------------

    def semantic(self, dim: int = 300, **kwargs) -> "Semantic":
        """Dense word vectors from this map's counts (PPMI + SVD); see `Semantic.from_map`."""
        from .semantic import Semantic

        return Semantic.from_map(self, dim=dim, **kwargs)

    def codec(self, **kwargs) -> "Codec":
        """A `Codec` over this map's vocabulary; keyword arguments go to `Codec`."""
        from .codec import Codec

        return Codec(self._words, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, map_dir: PathLike) -> None:
        """Write the map to `map_dir` (vocabulary.json + sums.npz)."""
        path = Path(map_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "vocabulary.json").write_text(
            json.dumps(
                {"format": _FORMAT_VERSION, "vocabulary": self.vocabulary},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        sp.save_npz(path / "sums.npz", self._sums.tocsr())

    @classmethod
    def load(cls, map_dir: PathLike) -> "LogPrime":
        """Load a map written by `save`."""
        path = Path(map_dir)
        vocab_path, sums_path = path / "vocabulary.json", path / "sums.npz"
        if not vocab_path.exists() or not sums_path.exists():
            raise FileNotFoundError(
                f"'{map_dir}' is not a logprime map (missing vocabulary.json or sums.npz)."
            )
        meta = json.loads(vocab_path.read_text("utf-8"))
        if meta.get("format") != _FORMAT_VERSION:
            raise ValueError(f"Unsupported map format in '{map_dir}'. Rebuild it from the corpus.")
        lp = cls()
        lp.vocabulary = {w: int(p) for w, p in meta["vocabulary"].items()}
        lp._words = sorted(lp.vocabulary, key=lp.vocabulary.get)
        lp._primes = [lp.vocabulary[w] for w in lp._words]
        lp._position = {w: i for i, w in enumerate(lp._words)}
        lp._next_prime = primes_after(lp._primes[-1] if lp._primes else 1)
        lp._sums = sp.load_npz(sums_path).astype(np.int64).tocsr()
        return lp

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:
        return f"LogPrime({len(self)} words, {self._sums.nnz} entries)"
