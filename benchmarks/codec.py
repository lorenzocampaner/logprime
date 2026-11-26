"""
Codec benchmark: exact decoding of real sentences, by length.

    python benchmarks/codec.py texts/ [--n 300] [--d 1024] [--k 8] [--max-n 5] [--positions 8]
    python benchmarks/codec.py texts/ --sweep        # words-only codec, several d and k
    python benchmarks/codec.py texts/ --bags         # random bags, to find the length limit

Each sentence is encoded and decoded back; the result counts as correct only if
it is certified (re-encodes to the same code) and equals the original.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

from logprime import Codec
from logprime.parser import parse_dir, parse_file

BUCKETS = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 10**9)]


def label(b):
    return f"{b[0]}-{b[1]}" if b[1] < 10**9 else f"{b[0]}+"


def run(codec, sents):
    tally, t = Counter(), 0.0
    for s in sents:
        code = codec.encode(s)
        t0 = time.perf_counter()
        r = codec.decode(code)
        t += time.perf_counter() - t0
        if r.status == "exact" and r.words != s or r.status == "bag" and sorted(r.words) != sorted(s):
            tally["wrong"] += 1  # a certified answer that is not the input: must never happen
        else:
            tally[r.status] += 1
    return tally, 1000 * t / max(len(sents), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--n", type=int, default=300, help="sentences per length bucket")
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=5, help="longest n-gram channel (1 = words only)")
    ap.add_argument("--positions", type=int, default=8, help="position slot size (0 = none)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--bags", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    sentences = [s.split() for s in (parse_dir(src) if src.is_dir() else parse_file(src))]
    vocab = list(dict.fromkeys(t for s in sentences for t in s))
    rng = random.Random(1)
    groups = {}
    for b in BUCKETS:
        pool = [s for s in sentences if b[0] <= len(s) <= b[1]]
        if pool:
            groups[b] = rng.sample(pool, min(args.n, len(pool)))
    print(f"{len(sentences)} sentences, {len(vocab)} words\n")

    if args.bags:
        print(f"  random bags, words only; share decoded exactly (20 per length)\n")
        lengths = (100, 200, 300, 400, 500, 600, 700, 800, 1000)
        print(f"  {'d':>5s} {'k':>3s}  " + "".join(f"{L:>6d}" for L in lengths))
        for d, k in ((1024, 8), (1024, 4), (1023, 3), (4096, 8)):
            codec = Codec(vocab, d=d, k=k, orders=())
            cells = [run(codec, [rng.choices(vocab, k=L) for _ in range(20)])[0]["bag"] * 5 for L in lengths]
            print(f"  {d:5d} {k:3d}  " + "".join(f"{c:5d}%" for c in cells), flush=True)
        return

    if args.sweep:
        print("  words only: share of sentences whose bag of words is decoded exactly\n")
        print(f"  {'d':>5s} {'k':>3s}  " + "".join(f"{label(b):>9s}" for b in groups) + f"{'ms':>7s}")
        for d in (256, 512, 1024, 4096):
            for k in (4, 8):
                codec = Codec(vocab, d=d, k=k, orders=())
                cells, ms = [], []
                for sents in groups.values():
                    tally, t = run(codec, sents)
                    assert not tally["wrong"]
                    cells.append(f"{100 * tally['bag'] / len(sents):8.1f}%")
                    ms.append(t * len(sents))
                total = sum(len(s) for s in groups.values())
                print(f"  {d:5d} {k:3d}  " + "".join(cells) + f"{sum(ms) / total:7.2f}", flush=True)
        return

    codec = Codec(vocab, d=args.d, k=args.k, orders=range(2, args.max_n + 1), positions=args.positions)
    print(f"  {codec}\n")
    print(f"  {'length':>8s} {'n':>5s} {'exact':>8s} {'ambiguous':>10s} {'bag':>6s} {'fail':>6s} {'wrong':>6s} {'ms':>6s}")
    for b, sents in groups.items():
        tally, ms = run(codec, sents)
        pct = lambda key: f"{100 * tally[key] / len(sents):.1f}%"
        print(f"  {label(b):>8s} {len(sents):5d} {pct('exact'):>8s} {pct('ambiguous'):>10s} {pct('bag'):>6s} "
              f"{pct('fail'):>6s} {tally['wrong']:6d} {ms:6.1f}", flush=True)
    print("\n  exact = the only order that fits, equal to the original; ambiguous = several orders fit;")
    print("  bag = words right, order not recovered; wrong = certified but different (must be 0)")


if __name__ == "__main__":
    main()
