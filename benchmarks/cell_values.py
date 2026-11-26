"""
Does exact decoding depend on the primes? Swap the values of M and see.

    python benchmarks/cell_values.py texts/ [--n 200]

Every variant keeps the same sparsity pattern (who follows whom, plus the
self cell) and changes only the numbers stored in it.
"""

import argparse
import random
from pathlib import Path

import numpy as np

from logprime import LogPrime
from logprime.parser import parse_dir, parse_file


def variants(lp):
    S = lp._sums.tocsr()
    S.sort_indices()
    row = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    col = S.indices
    logp = np.log(np.asarray(lp._primes, dtype=np.float64))
    counts = S.data / np.asarray(lp._primes)[col]  # Σp_j / p_j = number of co-occurrences
    rng = np.random.default_rng(0)
    return {
        "ln(p_i·Σp_j)    LogPrime": None,
        "Σ ln(p_i·p_j)   product": counts * (logp[row] + logp[col]),
        "ln(1 + count)   no primes": np.log1p(counts),
        "1               presence only": np.ones_like(counts),
        "uniform(1, 2)   random": rng.uniform(1, 2, len(counts)),
    }


def set_values(lp, data):
    lp._matrix = None
    M = lp.matrix  # rebuild helpers (pattern, norms, caches) from the sums
    if data is not None:
        M.data = data.astype(np.float64)
        lp._row_norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel())
        lp._matrix_csc = M.tocsc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    src = Path(args.source)
    sentences = parse_dir(src) if src.is_dir() else parse_file(src)
    lp = LogPrime().add(sentences)
    rng = random.Random(1)
    sents = [s.split() for s in rng.sample([s for s in sentences if len(s.split()) >= 3], args.n)]
    bags = [rng.choices(lp.words, k=rng.randint(3, 60)) for _ in range(args.n)]
    print(f"{len(sentences)} sentences, {len(lp)} words; {args.n} sentences and {args.n} random bags\n")
    print(f"  {'cell value':32s} {'sentences':>10s} {'bags':>8s}")
    for name, data in variants(lp).items():
        set_values(lp, data)
        res = []
        for items in (sents, bags):
            ok = 0
            for w in items:
                v = lp.vector(" ".join(w))
                out = lp.decode(v, n_words=len(w))
                ok += sorted(out) == sorted(w) and lp.verify(out, v)
            res.append(f"{100 * ok / len(items):.1f}%")
        print(f"  {name:32s} {res[0]:>10s} {res[1]:>8s}", flush=True)


if __name__ == "__main__":
    main()
