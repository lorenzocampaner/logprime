"""
Meaning read straight from the codes: the codec's meaning channel.

    python benchmarks/code_semantics.py texts/ wordsim353-italian.txt ... [--cache sem.npz]

`Codec(semantic=sem)` adds one row to every code: the sum of the words' unit
vectors, ×1000 and rounded. This script checks three things:

  meaning   Spearman ρ between human scores and `Codec.similarity` of two
            codes, next to `Semantic.similarity` of the same words (the
            rounding should cost nothing)
  decode    share of real sentences decoded exactly, order included, with and
            without the meaning row (decoding ignores it, so they must match)
  sums      the meaning row of a sum of codes equals the meaning row of the
            joined sentences, so arithmetic on codes stays meaningful
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from similarity import evaluate, read_pairs  # noqa: E402

from logprime import LogPrime, Semantic  # noqa: E402
from logprime.parser import parse_dir, parse_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--n", type=int, default=300, help="sentences for the decoding check")
    ap.add_argument("--cache", help=".npz file to save/load the word vectors")
    args = ap.parse_args()

    src = Path(args.source)
    sentences = parse_dir(src) if src.is_dir() else parse_file(src)
    lp = LogPrime().add(sentences)
    if args.cache and Path(args.cache).exists():
        sem = Semantic.load(args.cache)
    else:
        sem = lp.semantic(300)
        if args.cache:
            sem.save(args.cache)
    plain, codec = lp.codec(), lp.codec(semantic=sem)
    print(f"{len(sentences)} sentences, {len(lp)} words; {codec}\n")

    print(f"  {'':24s}" + "".join(f"{Path(p).stem.replace('-italian', ''):>18s}" for p in args.datasets))
    for name, sim in (("Semantic.similarity", sem.similarity), ("Codec.similarity", codec.similarity)):
        cells = []
        for path in args.datasets:
            rho, p, n = evaluate(sim, lp._position, read_pairs(path))
            cells.append(f"{rho:+.3f}{'*' if p < 0.05 else ' '} ({n})")
        print(f"  {name:24s}" + "".join(f"{c:>18s}" for c in cells), flush=True)

    rng = random.Random(1)
    sample = rng.sample(sentences, min(args.n, len(sentences)))
    exact = [sum(c.decode(c.encode(s)).text == s for s in sample) for c in (plain, codec)]
    print(f"\n  decoded exactly, {len(sample)} sentences: without meaning {exact[0]}, with meaning {exact[1]}")
    pairs = [rng.sample(sentences, 2) for _ in range(100)]
    same = sum(np.array_equal((codec.encode(a) + codec.encode(b))[-1], codec.encode(a + " " + b)[-1])
               for a, b in pairs)
    print(f"  meaning of a sum of two codes = meaning of the joined sentences: {same}/100")
    print("\n  ρ: Spearman, * = p < 0.05, (pairs covered)")


if __name__ == "__main__":
    main()
