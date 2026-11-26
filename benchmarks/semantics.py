"""
Word meaning from the same count map.

    python benchmarks/semantics.py texts/ simlex-italian.txt wordsim353-italian.txt ...

`Semantic.from_map` (PPMI + SVD on the co-occurrence counts the map already
holds) compared with the same model built from a ±5-word window, which needs
the text again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).parent))
from similarity import evaluate, ppmi, read_pairs, row_normalize, vector_sim, window_counts  # noqa: E402

from logprime import LogPrime, Semantic  # noqa: E402
from logprime.parser import parse_dir, parse_file  # noqa: E402
from logprime.semantic import cooccurrence  # noqa: E402


def svd_embedding(P, dim=300, power=0.5):
    U, s, _ = svds(P, k=dim, v0=np.random.default_rng(0).standard_normal(min(P.shape)))
    return row_normalize(U * s**power)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--probe", default="re,guerra,mare,occhi,partito,casa,notte,donna")
    args = ap.parse_args()

    src = Path(args.source)
    sentences = parse_dir(src) if src.is_dir() else parse_file(src)
    lp = LogPrime().add(sentences)
    index = lp._position
    print(f"{len(sentences)} sentences, {len(lp)} words\n")

    sem = Semantic.from_map(lp, dim=300)
    methods = {
        "PPMI, map counts": vector_sim(row_normalize(ppmi(cooccurrence(lp))), index),
        "Semantic.from_map (PPMI+SVD 300)": sem.similarity,
    }
    P = ppmi(window_counts(lp, sentences))
    methods["PPMI, window ±5 (re-read text)"] = vector_sim(row_normalize(P), index)
    methods["PPMI+SVD 300, window ±5"] = vector_sim(svd_embedding(P, 300), index)

    datasets = [(Path(d).stem.replace("-italian", ""), read_pairs(d)) for d in args.datasets]
    print(f"  {'method':40s}" + "".join(f"{n:>18s}" for n, _ in datasets))
    for name, sim in methods.items():
        cells = []
        for _, pairs in datasets:
            rho, p, _ = evaluate(sim, index, pairs)
            cells.append(f"{rho:+.3f}{'*' if p < 0.05 else ' '}")
        print(f"  {name:40s}" + "".join(f"{c:>18s}" for c in cells), flush=True)
    print("\n  Spearman ρ, * = p < 0.05\n\n  nearest neighbours, Semantic.from_map (words seen ≥ 5 times):")
    for w in args.probe.split(","):
        if w in index:
            print(f"    {w:10s} {', '.join(x for x, _ in sem.neighbours(w, 8))}")


if __name__ == "__main__":
    main()
