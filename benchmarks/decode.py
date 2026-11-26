"""
Decoding benchmark: how often does decode() recover a sentence exactly, and how fast?

    python benchmarks/decode.py texts/ [--n 200] [--seed 1]

`texts/` is a .txt file or a directory of .txt files (parsed with the
built-in parser). Reports, by sentence length: exact recoveries, verified
results, wrong-but-verified results (should always be 0) and time per sentence.
Random bags of vocabulary words are tested as well.
"""

import argparse
import random
import time
from pathlib import Path

from logprime import LogPrime
from logprime.parser import parse_dir, parse_file

BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 80)]


def run(lp, items, label):
    exact = verified = wrong_verified = 0
    t0 = time.perf_counter()
    for words in items:
        vec = lp.vector(" ".join(words))
        out = lp.decode(vec, n_words=len(words))
        ok = sorted(out) == sorted(words)
        ver = lp.verify(out, vec)
        exact += ok
        verified += ver
        wrong_verified += ver and not ok
    ms = 1000 * (time.perf_counter() - t0) / max(len(items), 1)
    print(f"  {label:14s} {len(items):5d}  exact {100 * exact / len(items):5.1f}%  "
          f"verified {100 * verified / len(items):5.1f}%  wrong+verified {wrong_verified}  {ms:8.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--n", type=int, default=200, help="samples per bucket")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    src = Path(args.source)
    sentences = parse_dir(src) if src.is_dir() else parse_file(src)
    t0 = time.perf_counter()
    lp = LogPrime().add(sentences)
    lp.matrix
    print(f"{len(sentences)} sentences, {len(lp)} words, {lp.matrix.nnz} entries "
          f"(build {time.perf_counter() - t0:.1f}s)\n")

    rng = random.Random(args.seed)
    print("  sentences      count")
    for lo, hi in BUCKETS:
        pool = [s.split() for s in sentences if lo <= len(s.split()) <= hi]
        if pool:
            run(lp, rng.sample(pool, min(args.n, len(pool))), f"{lo}-{hi} words")
    print("\n  random bags of vocabulary words (with repeats)")
    for lo, hi in BUCKETS:
        run(lp, [rng.choices(lp.words, k=rng.randint(lo, hi)) for _ in range(args.n)], f"{lo}-{hi} words")


if __name__ == "__main__":
    main()
