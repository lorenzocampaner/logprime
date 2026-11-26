"""Prime number source used to assign a unique prime to each new word."""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np


def _sieve(limit: int) -> np.ndarray:
    """All primes < limit."""
    if limit < 3:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit, dtype=bool)
    is_prime[:2] = False
    for p in range(2, math.isqrt(limit - 1) + 1):
        if is_prime[p]:
            is_prime[p * p :: p] = False
    return np.flatnonzero(is_prime)


def primes_after(n: int) -> Iterator[int]:
    """Yield every prime strictly greater than n, in increasing order.

    Uses a segmented sieve, so resuming after a large prime is cheap.
    """
    lo = n + 1
    size = max(1024, lo)
    while True:
        hi = lo + size
        base = _sieve(math.isqrt(hi) + 1)
        is_prime = np.ones(hi - lo, dtype=bool)
        for p in base:
            start = max(p * p, -(-lo // p) * p)
            is_prime[start - lo :: p] = False
        if lo < 2:
            is_prime[: 2 - lo] = False
        for k in np.flatnonzero(is_prime):
            yield int(lo + k)
        lo = hi
