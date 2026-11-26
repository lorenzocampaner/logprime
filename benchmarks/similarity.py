"""
Word-similarity benchmark: how well does each word vector agree with human judgements?

    python benchmarks/similarity.py texts/ simlex-italian.txt wordsim353-italian.txt ...

Each dataset is a tab-separated file `word1<TAB>word2<TAB>score` with one
header line, e.g. the Italian SimLex-999 and WordSim-353 of Leviant & Reichart
(2015): https://github.com/iraleviant/eval-multilingual-simlex

The score is Spearman's ρ between human scores and cosine similarity, over the
pairs whose two words are both in the corpus vocabulary. All methods are built
from the same parsed sentences, so they see the same pairs.

Methods:
  LogPrime similarity() the library's own measure: both rows projected onto
                        the columns of the two words, [M[a][a], M[a][b]] vs
                        [M[b][a], M[b][b]]
  presence projection   the same with every value set to 1 (no primes)
  LogPrime full row     plain cosine of the whole rows M[a] and M[b]
  PPMI                  classic count model: ±5-word window, positive PMI
  PPMI + SVD            the same, reduced to 300 dimensions
  word2vec              skip-gram, 300 dims (only if gensim is installed)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.stats import spearmanr

from logprime import LogPrime
from logprime.parser import parse_dir, parse_file


def read_pairs(path):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        a, b, s = line.split("\t")[:3]
        pairs.append((a.strip().lower(), b.strip().lower(), float(s)))
    return pairs


def row_normalize(X):
    if sp.issparse(X):
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1
        return sp.diags(1 / norms) @ X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def window_counts(lp, sentences, window=5):
    pos = lp._position
    rows, cols = [], []
    for s in sentences:
        ids = np.array([pos[t] for t in s.split()])
        for d in range(1, window + 1):
            rows += [ids[:-d], ids[d:]]
            cols += [ids[d:], ids[:-d]]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    n = len(lp)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def ppmi(C, alpha=0.75):
    C = C.tocoo()
    total = C.sum()
    w = np.asarray(C.sum(axis=1)).ravel()
    c = np.asarray(C.sum(axis=0)).ravel() ** alpha
    c = c / c.sum()
    pmi = np.log(C.data * 1.0 / total) - np.log(w[C.row] / total) - np.log(c[C.col])
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)


def vector_sim(vectors, index):
    """Cosine similarity function for a row-normalized matrix."""
    def sim(a, b):
        va, vb = vectors[index[a]], vectors[index[b]]
        return va.multiply(vb).sum() if sp.issparse(vectors) else float(va @ vb)
    return sim


def projected_sim(M, index):
    """LogPrime.similarity() for two words, on an arbitrary matrix M."""
    def sim(a, b):
        i, j = index[a], index[b]
        va = np.array([M[i, i], M[i, j]])
        vb = np.array([M[j, i], M[j, j]])
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))
    return sim


def evaluate(sim, index, pairs):
    human, model = [], []
    for a, b, s in pairs:
        if a in index and b in index:
            human.append(s)
            model.append(sim(a, b))
    r = spearmanr(human, model)
    return r.statistic, r.pvalue, len(human)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("datasets", nargs="+")
    args = ap.parse_args()

    src = Path(args.source)
    sentences = parse_dir(src) if src.is_dir() else parse_file(src)
    lp = LogPrime().add(sentences)
    M = lp.matrix.tocsr()
    index = lp._position
    tokens = sum(len(s.split()) for s in sentences)
    print(f"{len(sentences)} sentences, {tokens} tokens, {len(lp)} words\n")

    methods = {}
    t = time.time()
    methods["LogPrime similarity()"] = lambda a, b: lp.similarity(a, b)
    pres = M.copy()
    pres.data[:] = 1
    methods["presence projection"] = projected_sim(pres, index)
    methods["LogPrime full row"] = vector_sim(row_normalize(M), index)
    P = ppmi(window_counts(lp, sentences))
    methods["PPMI (window 5)"] = vector_sim(row_normalize(P), index)
    U, S, _ = svds(P, k=300)
    methods["PPMI + SVD 300"] = vector_sim(row_normalize(U * np.sqrt(S)), index)
    print(f"count-based models built in {time.time() - t:.0f}s")

    try:
        from gensim.models import Word2Vec  # gensim 4.3 needs numpy<2, scipy<1.14

        t = time.time()
        w2v = Word2Vec([s.split() for s in sentences], vector_size=300, window=5,
                       min_count=1, sg=1, negative=10, epochs=10, workers=4, seed=1)
        W = np.zeros((len(lp), 300))
        for w, i in index.items():
            W[i] = w2v.wv[w]
        methods["word2vec skip-gram 300"] = vector_sim(row_normalize(W), index)
        print(f"word2vec trained in {time.time() - t:.0f}s")
    except ImportError:
        print("gensim not installed: skipping word2vec")

    datasets = [(Path(d).stem, read_pairs(d)) for d in args.datasets]
    print()
    header = f"  {'method':26s}" + "".join(f"{name:>28s}" for name, _ in datasets)
    print(header)
    for name, sim in methods.items():
        cells = []
        for _, pairs in datasets:
            rho, pval, n = evaluate(sim, index, pairs)
            star = "*" if pval < 0.05 else " "
            cells.append(f"{rho:+.3f}{star} ({n}/{len(pairs)})")
        print(f"  {name:26s}" + "".join(f"{c:>28s}" for c in cells), flush=True)
    print("\n  ρ = Spearman correlation, * = p < 0.05, (pairs covered / total)")


if __name__ == "__main__":
    main()
