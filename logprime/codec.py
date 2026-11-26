"""
Codec: fixed-size sentence codes that decode back exactly, word order included.

A sentence becomes a small integer matrix of d columns, one row per channel:

  row 0      word channel: the sum of each word's code. A word's code lights
             one cell in each of k blocks of d/k cells, with its prime as the
             value: the i-th word of the vocabulary gets the i-th prime, as in
             a LogPrime map, so no two words ever have the same value.
  row 1, …   one row per n-gram order n in `orders`: the sum of a code for
             every run of n adjacent tokens of  <s> w1 … wL </s>.
  next row   position channel (if `positions` > 0): the sum of a code for
             every pair (word, its position // positions).
  last row   meaning channel (with a `Semantic` model): the sum of the words'
             vectors, rounded to integers, in its first `dim` cells.

Cells come from a hash of the word (n-gram codes from a hash of the n-gram), so
nothing has to be stored per word, and codes do not change when new words are
appended to the vocabulary. Every row is a sum over the words (or n-grams), so
codes can be added and subtracted. The meaning row makes `similarity` a cosine
of meanings, read straight from two codes without decoding them; decoding
ignores it, and verification checks it like every other row.

Decoding, channel by channel:
  1. word channel → the bag of words. A cell holding a single word holds n·p,
     so factoring it gives the word and its count; confirmed words are
     subtracted, which frees more cells (peeling). What is left is solved by
     least squares: a word can only be present if all its k cells are
     non-zero, and the fit on those candidates, rounded, gives the counts.
  2. the only possible bigrams are pairs of decoded words (plus <s>, </s>); the
     only possible n-grams are overlapping (n-1)-grams. Each channel is solved
     the same way among its candidates.
  3. the position channel is solved among (decoded word, slot) pairs;
  4. the sentence is a walk from <s> to </s> that uses every top-order n-gram
     exactly once (an Eulerian path) and puts every word in a slot the
     position channel allows.

Every answer is re-encoded and compared exactly with the input, so a decode is
either certified or reported as not certified; it is never silently wrong.
Without the position channel, order can be ambiguous: when a stretch of at
least n-1 words repeats, the parts between the repeats can be swapped without
changing any n-gram. The position channel tells those orders apart, because
the swapped parts land in different slots.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.linalg

from .primes import primes_after

if TYPE_CHECKING:
    from .semantic import Semantic

PathLike = Union[str, Path]
Tokens = Union[str, Sequence[str]]

_FORMAT_VERSION = 2
_MEANING_SCALE = 1000  # unit vectors × 1000, rounded: integer meaning rows
_VMAX = 1 << 20
_PEEL_MAX = 16  # largest count of one word read off a single cell
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def _first_primes(n: int) -> List[int]:
    it = primes_after(1)
    return [next(it) for _ in range(n)]


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, vectorized on uint64."""
    with np.errstate(over="ignore"):
        x = (x + np.uint64(0x9E3779B97F4A7C15)) & _MASK
        x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _MASK
        x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _MASK
        return x ^ (x >> np.uint64(31))


def _word_key(word: str) -> int:
    return int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "little")


_BOS_KEY = _word_key("\x00<s>")
_EOS_KEY = _word_key("\x00</s>")


def _tokens(text: Tokens) -> List[str]:
    return text.lower().split() if isinstance(text, str) else [t.lower() for t in text]


@dataclass
class Decoded:
    """Result of `Codec.decode`.

    status
      "exact"      the word order was recovered, it is the only one that fits,
                   and it reproduces the code exactly;
      "ambiguous"  several orders reproduce the code exactly; `words` is one of
                   them and `alternatives` holds the others found;
      "bag"        the words are certified, the order is not (no n-gram
                   channels, or a code that is a sum of several sentences):
                   `words` is the multiset in no particular order;
      "fail"       nothing could be certified; `words` is empty.
    """

    status: str
    words: List[str]
    alternatives: List[List[str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.status != "fail"

    @property
    def text(self) -> str:
        return " ".join(self.words)


class Codec:
    """Encode sentences as fixed-size integer codes and decode them exactly.

    vocabulary : the words that can be decoded (e.g. `LogPrime.words`).
    d          : cells per channel; must be a multiple of k.
    k          : cells lit by each word (one per block of d/k cells).
    orders     : n-gram channels used to recover the order, e.g. (2, 3, 4, 5);
                 () keeps only the bag of words.
    positions  : slot size of the position channel (default 8: each word is
                 tagged with position // 8); 0 leaves it out. It makes the
                 order exact where n-grams alone are ambiguous. Ignored
                 when `orders` is empty.
    seed       : selects the hash family; same seed, same codes.
    semantic   : optional `Semantic` model (or any with `.index` and
                 `.vectors`). Adds the meaning channel: one more row, the sum
                 of the words' unit vectors scaled by 1000 and rounded, so
                 `similarity` compares meanings. Its dim must be at most d.
    """

    def __init__(
        self,
        vocabulary: Iterable[str],
        d: int = 1024,
        k: int = 8,
        orders: Sequence[int] = (2, 3, 4, 5),
        positions: int = 8,
        seed: int = 0,
        semantic: Optional["Semantic"] = None,
    ) -> None:
        orders = sorted(set(int(n) for n in orders))
        if d % k:
            raise ValueError(f"d ({d}) must be a multiple of k ({k}).")
        if orders and orders != list(range(2, orders[-1] + 1)):
            raise ValueError(f"orders must be 2, 3, … without gaps, got {orders}.")
        if positions < 0:
            raise ValueError(f"positions must be >= 0, got {positions}.")
        if not orders:
            positions = 0  # slots only help to order n-grams
        self.words: List[str] = list(dict.fromkeys(w.lower() for w in vocabulary))
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.words)}
        self.d, self.k, self.orders, self.seed = d, k, tuple(orders), seed
        self.positions = int(positions)
        self._keys = np.array([_word_key(w) for w in self.words], dtype=np.uint64)
        self._pos, self._val = self._word_codes()
        self._meaning = None if semantic is None else self._meaning_codes(semantic)
        self._set_shape()

    def _set_shape(self) -> None:
        rows = 1 + len(self.orders)
        self._slot_row = rows if self.positions else None
        rows += 1 if self.positions else 0
        self._meaning_row = rows if self._meaning is not None else None
        rows += 1 if self._meaning is not None else 0
        self.shape = (rows, self.d)

    @property
    def semantic(self) -> bool:
        """True if the codes carry the meaning channel."""
        return self._meaning is not None

    # ------------------------------------------------------------------
    # Codes
    # ------------------------------------------------------------------

    def _word_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        V, k, b = len(self.words), self.k, self.d // self.k
        with np.errstate(over="ignore"):
            h = _mix(self._keys[:, None] ^ _mix(np.arange(k, dtype=np.uint64) + np.uint64(self.seed))[None, :])
        cell = (h % np.uint64(b)).astype(np.int64)
        # Each word's value is its prime: the i-th word gets the i-th prime,
        # the same one it has in a LogPrime map built from the same text.
        val = np.repeat(np.array(_first_primes(V), dtype=np.int64)[:, None], k, axis=1)
        return cell + np.arange(k) * b, val

    def _meaning_codes(self, semantic: "Semantic") -> np.ndarray:
        """Each word's vector, unit length, times _MEANING_SCALE, rounded (V × dim)."""
        missing = [w for w in self.words if w not in semantic.index]
        if missing:
            raise ValueError(f"{len(missing)} words have no semantic vector, e.g. {missing[:3]}")
        E = np.asarray(semantic.vectors, dtype=np.float64)[[semantic.index[w] for w in self.words]]
        if E.shape[1] > self.d:
            raise ValueError(f"semantic vectors have {E.shape[1]} dimensions, more than d ({self.d}).")
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return np.rint(_MEANING_SCALE * E / norms).astype(np.int64)

    def _ngram_codes(self, keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cells and values (m × k) of the n-grams whose word keys are the rows of `keys`."""
        m, n = keys.shape
        with np.errstate(over="ignore"):
            h = np.full(m, _mix(np.uint64(n) * np.uint64(0x100000001B3) + np.uint64(self.seed)))
            for c in range(n):
                h = _mix(h ^ keys[:, c])
            h = _mix(h[:, None] + np.arange(self.k, dtype=np.uint64)[None, :] * np.uint64(0x9E3779B1))
        pos = (h % np.uint64(self.d)).astype(np.int64)
        val = (_mix(h) % np.uint64(_VMAX - 1)).astype(np.int64) + 1
        return pos, val

    def _slot_codes(self, ids: np.ndarray, slots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cells and values (m × k) of the position-channel codes of (word, slot) pairs."""
        with np.errstate(over="ignore"):
            slot_keys = _mix(slots.astype(np.uint64) + np.uint64(0x51E7_0000_0000_0000))
        return self._ngram_codes(np.stack((self._keys[ids], slot_keys), axis=1))

    def _key_seq(self, ids: Sequence[int]) -> np.ndarray:
        ends = np.array([_BOS_KEY, _EOS_KEY], dtype=np.uint64)
        return np.concatenate((ends[:1], self._keys[list(ids)], ends[1:]))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def ids(self, text: Tokens) -> List[int]:
        tokens = _tokens(text)
        unknown = [t for t in tokens if t not in self._index]
        if unknown:
            raise ValueError(f"not in the vocabulary: {' '.join(dict.fromkeys(unknown))}")
        return [self._index[t] for t in tokens]

    def encode(self, text: Tokens) -> np.ndarray:
        """Code of a sentence: int64 array of shape `self.shape`,
        (1 + len(orders) + 1 if positions, d).

        Codes of several sentences can be added; the sum still decodes to the
        bag of all their words (status "bag").
        """
        ids = self.ids(text)
        code = np.zeros(self.shape, dtype=np.int64)
        np.add.at(code[0], self._pos[ids].ravel(), self._val[ids].ravel())
        seq = self._key_seq(ids)
        for r, n in enumerate(self.orders, 1):
            if len(seq) >= n:
                grams = np.lib.stride_tricks.sliding_window_view(seq, n)
                pos, val = self._ngram_codes(grams)
                np.add.at(code[r], pos.ravel(), val.ravel())
        if self.positions and ids:
            pos, val = self._slot_codes(np.array(ids), np.arange(len(ids)) // self.positions)
            np.add.at(code[self._slot_row], pos.ravel(), val.ravel())
        if self._meaning is not None and ids:
            m = self._meaning.shape[1]
            code[self._meaning_row, :m] = self._meaning[ids].sum(axis=0)
        return code

    def verify(self, words: Tokens, code: np.ndarray, channels: Optional[int] = None) -> bool:
        """True if `words` (in this order) encode to exactly `code`.

        channels : compare only the first `channels` rows (1 = the bag only).
        """
        try:
            mine = self.encode(words)
        except ValueError:
            return False
        rows = slice(0, channels)
        return bool(np.array_equal(mine[rows], np.asarray(code)[rows]))

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    @staticmethod
    def _solve(pos: np.ndarray, val: np.ndarray, v: np.ndarray) -> Optional[np.ndarray]:
        """Integer counts x ≥ 0 of the candidate codes (pos, val) that sum to v.

        Returns counts for every candidate (0 for absent ones), or None if no
        consistent solution was found.
        """
        present = v > 0
        keep = present[pos].all(axis=1)
        counts = np.zeros(len(pos), dtype=np.int64)
        rows = np.flatnonzero(present)
        cand = np.flatnonzero(keep)
        if len(cand) == 0:
            return counts if len(rows) == 0 else None
        if len(cand) > len(rows):
            return None
        row_of = np.full(len(v), -1)
        row_of[rows] = np.arange(len(rows))
        A = np.zeros((len(rows), len(cand)))
        for j, c in enumerate(cand):
            np.add.at(A[:, j], row_of[pos[c]], val[c])
        b = v[rows].astype(np.float64)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        y = np.rint(x)
        if (y < 0).any() or not np.array_equal(A @ y, b):
            y = Codec._integer_point(A, b, x)
            if y is None:
                y = np.maximum(np.rint(x), 0)  # not exact: verification will report it
        counts[cand] = y.astype(np.int64)
        return counts

    @staticmethod
    def _integer_point(A: np.ndarray, b: np.ndarray, x: np.ndarray,
                       max_tries: int = 4096) -> Optional[np.ndarray]:
        """Non-negative integer y with A y = b, near the least-squares x.

        When candidate codes overlap, A can lose rank: over the reals the
        solutions form a line or plane x + N t, and least squares picks a point
        on it that is not the integer one. The integer points on it are far
        apart, so the true counts are, in practice, the only small
        non-negative integer point. It is found by trying small counts for q
        pivot entries.
        """
        _, sv, vt = np.linalg.svd(A)
        tol = sv.max() * max(A.shape) * np.finfo(float).eps
        N = vt[int((sv > tol).sum()):].T
        q = N.shape[1]
        if q == 0 or q > 3:
            return None
        piv = scipy.linalg.qr(N.T, pivoting=True)[2][:q]
        with np.errstate(divide="ignore", invalid="ignore"):
            ub = np.where(A > 0, b[:, None] / A, np.inf).min(axis=0)
        ranges = [range(int(min(ub[i], 64)) + 1) for i in piv]
        if np.prod([len(r) for r in ranges], dtype=float) > max_tries:
            return None
        for m in itertools.product(*ranges):
            t = np.linalg.solve(N[piv], np.array(m, dtype=float) - x[piv])
            y = np.rint(x + N @ t)
            if (y >= 0).all() and np.array_equal(A @ y, b):
                return y
        return None

    def _peel(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Read words straight off the cells, by factoring their values.

        A cell that holds a single word holds n·p: n occurrences of the word
        whose prime is p. Factoring the value gives the word and its count
        with no search. Such a reading is accepted only if none of the word's
        other cells holds less than n·p; the word is then subtracted, which can leave more cells with a single
        word. Returns (counts read, residual).
        """
        r = v.astype(np.int64).copy()
        counts = np.zeros(len(self.words), dtype=np.int64)
        primes = self._val[:, 0]  # increasing: index = word id
        while r.any():
            seen = {}
            for n in range(1, _PEEL_MAX + 1):
                cells = np.flatnonzero((r > 0) & (r % n == 0))
                q = r[cells] // n
                i = np.minimum(np.searchsorted(primes, q), len(primes) - 1)
                for w in i[primes[i] == q]:
                    seen.setdefault(int(w), n)
            progress = False
            for w, n in seen.items():
                value = n * primes[w]
                here = r[self._pos[w]]
                if (here >= value).all() and (here == value).any():
                    r[self._pos[w]] -= value
                    counts[w] += n
                    progress = True
            if not progress:
                break
        return counts, r

    def _decode_bag(self, v: np.ndarray) -> Optional[List[int]]:
        counts, rest = self._peel(v)
        if rest.any():
            more = self._solve(self._pos, self._val, rest)
            if more is None or not np.array_equal(
                    np.bincount(self._pos.ravel(), (self._val * (counts + more)[:, None]).ravel(),
                                minlength=self.d).astype(np.int64), v):
                more, counts = self._solve(self._pos, self._val, v), 0  # start over without peeling
                if more is None:
                    return None
            counts = counts + more
        return [i for i in np.flatnonzero(counts) for _ in range(counts[i])]

    def _decode_slots(self, bag: List[int], v: np.ndarray) -> Optional[Dict[tuple, int]]:
        """How many times each word sits in each slot, from the position channel."""
        nodes = np.array(sorted(set(bag)), dtype=np.int64)
        n_slots = -(-len(bag) // self.positions)
        ids, slots = np.repeat(nodes, n_slots), np.tile(np.arange(n_slots), len(nodes))
        counts = self._solve(*self._slot_codes(ids, slots), v)
        if counts is None:
            return None
        return {(int(i), int(t)): int(c) for i, t, c in zip(ids, slots, counts) if c}

    def _walks(self, grams: Dict[tuple, int], limit: int,
               slots: Optional[Dict[tuple, int]] = None, budget: int = 200_000):
        """Up to `limit` id sequences using every n-gram of `grams` exactly once.

        With `slots`, the i-th word must be one that the position channel puts
        in slot i // positions.
        """
        slots = dict(slots) if slots is not None else None
        B = self.positions or 1

        def take(word, i):  # claim slot i for word; False if the channel forbids it
            if slots is None or word < 0:
                return True
            key = (word, i // B)
            if slots.get(key, 0) == 0:
                return False
            slots[key] -= 1
            return True

        def give(word, i):
            if slots is not None and word >= 0:
                slots[(word, i // B)] += 1

        out: Dict[tuple, Dict[tuple, int]] = {}
        for g, c in grams.items():
            out.setdefault(g[:-1], {})[g] = c
        total = sum(grams.values())
        starts = [g for g in grams if g[0] == -1]
        found: List[List[int]] = []
        steps = 0

        def dfs(node, path, used):
            nonlocal steps
            steps += 1
            if steps > budget or len(found) >= limit:
                return
            if used == total:
                if path[-1] == -2:
                    found.append([i for i in path if i >= 0])
                return
            i = len(path) - 1  # position of the next word (path starts with <s>)
            for g, c in out.get(node, {}).items():
                if c and take(g[-1], i):
                    out[node][g] = c - 1
                    path.append(g[-1])
                    dfs(g[1:], path, used + 1)
                    path.pop()
                    out[node][g] = c
                    give(g[-1], i)

        for g in starts:
            words = [(w, i) for i, w in enumerate(g[1:])]
            taken = []
            for w, i in words:
                if not take(w, i):
                    break
                taken.append((w, i))
            else:
                out[g[:-1]][g] -= 1
                dfs(g[1:], list(g), 1)
                out[g[:-1]][g] += 1
            for w, i in taken:
                give(w, i)
        return found, steps > budget

    def _decode_order(self, bag: List[int], code: np.ndarray) -> Optional[Dict[tuple, int]]:
        """Counts of the top-order n-grams, as tuples of ids (-1 = <s>, -2 = </s>)."""
        nodes = sorted(set(bag))
        ext = np.concatenate((self._keys, np.array([_BOS_KEY, _EOS_KEY], dtype=np.uint64)))
        V = len(self._keys)
        left, right = [-1] + nodes, nodes + [-2]
        cand = np.array([(a, b) for a in left for b in right], dtype=np.int64).reshape(-1, 2)
        top: Dict[tuple, int] = {}
        for r, n in enumerate(self.orders, 1):
            if len(cand) == 0:
                break
            keys = ext[np.where(cand >= 0, cand, V - 1 - cand)]  # -1 → <s>, -2 → </s>
            counts = self._solve(*self._ngram_codes(keys), code[r])
            if counts is None:
                return None
            grams = {tuple(int(t) for t in g): int(c) for g, c in zip(cand, counts) if c}
            if not grams:
                break
            top = grams
            by_prefix: Dict[tuple, List[tuple]] = {}
            for g in grams:
                by_prefix.setdefault(g[:-1], []).append(g)
            cand = np.array(
                [g + (h[-1],) for g in grams for h in by_prefix.get(g[1:], [])], dtype=np.int64
            ).reshape(-1, n + 1)
        return top

    def decode(self, code: np.ndarray, max_orders: int = 2) -> Decoded:
        """Recover the sentence (or the bag of words) that `code` encodes.

        max_orders : how many distinct orders to look for before calling the
                     result ambiguous (2 is enough to tell).
        """
        code = np.asarray(code)
        if code.shape != self.shape:
            raise ValueError(f"expected a code of shape {self.shape}, got {code.shape}")
        bag = self._decode_bag(code[0])
        if bag is None or not self.verify([self.words[i] for i in bag], code, channels=1):
            return Decoded("fail", [])
        bag_words = [self.words[i] for i in bag]
        if not self.orders:
            return Decoded("bag", bag_words)
        top = self._decode_order(bag, code)
        slots = self._decode_slots(bag, code[self._slot_row]) if self.positions and bag else None
        if top is not None:
            walks, gave_up = self._walks(top, max_orders, slots)
            orders = [[self.words[i] for i in w] for w in walks]
            orders = [w for w in orders if self.verify(w, code)]
            if len(orders) == 1 and not gave_up:
                return Decoded("exact", orders[0])
            if orders:
                return Decoded("ambiguous", orders[0], orders[1:])
        return Decoded("bag", bag_words)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def similarity(self, a: Union[Tokens, np.ndarray], b: Union[Tokens, np.ndarray]) -> float:
        """Similarity of two texts or codes, read from the codes alone.

        With the meaning channel: the cosine of the two meaning rows, i.e. of
        the sums of the words' vectors (as `Semantic.similarity`, up to the
        rounding). Without it: the cosine of the word channels' supports
        (which cells are lit), which only measures shared words.
        """
        ca = self.encode(a) if not isinstance(a, np.ndarray) else a
        cb = self.encode(b) if not isinstance(b, np.ndarray) else b
        if self._meaning is not None:
            va = ca[self._meaning_row].astype(np.float64)
            vb = cb[self._meaning_row].astype(np.float64)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))
        va, vb = ca[0] > 0, cb[0] > 0
        na, nb = va.sum(), vb.sum()
        return 0.0 if na == 0 or nb == 0 else float((va & vb).sum() / np.sqrt(na * nb))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: PathLike) -> None:
        """Write the codec to a directory (codec.json, plus meaning.npy if semantic)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"format": _FORMAT_VERSION, "d": self.d, "k": self.k, "orders": list(self.orders),
                "positions": self.positions,
                "seed": self.seed, "semantic": self.semantic, "vocabulary": self.words}
        (path / "codec.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        if self.semantic:
            np.save(path / "meaning.npy", self._meaning)

    @classmethod
    def load(cls, path: PathLike) -> "Codec":
        path = Path(path)
        meta_path = path / "codec.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"'{path}' has no codec.json")
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("format") != _FORMAT_VERSION:
            raise ValueError(f"Unsupported codec format in '{path}'.")
        codec = cls(meta["vocabulary"], d=meta["d"], k=meta["k"], orders=meta["orders"],
                    positions=meta.get("positions", 0), seed=meta["seed"])
        if meta["semantic"]:
            codec._meaning = np.load(path / "meaning.npy")
            codec._set_shape()
        return codec

    def __len__(self) -> int:
        return len(self.words)

    def __repr__(self) -> str:
        meaning = f", meaning dim={self._meaning.shape[1]}" if self.semantic else ""
        return (f"Codec({len(self)} words, d={self.d}, k={self.k}, orders={self.orders}, "
                f"positions={self.positions}{meaning})")
