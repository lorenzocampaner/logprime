"""
Command-line interface.

    logprime build corpus.txt -o map/            # one sentence per line
    logprime build texts/ -o map/ --parse        # raw text, split and cleaned
    logprime build more.txt -o map/ --add        # add text to an existing map
    logprime info map/ [--word WORD]
    logprime decode map/ "la guerra continua senza fine"
    logprime code map/ "la guerra continua senza fine"     # fixed-size code, order included
    logprime semantic map/ -w guerra -w notte             # nearest words (PPMI + SVD)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from .core import LogPrime
from .semantic import Semantic
from .parser import STOPWORDS_EN, STOPWORDS_IT, parse_text

STOPWORDS = {"it": STOPWORDS_IT, "en": STOPWORDS_EN}


def _text_files(source: Path) -> List[Path]:
    if source.is_dir():
        files = sorted(source.glob("*.txt"))
        if not files:
            sys.exit(f"[error] no .txt files in '{source}'")
        return files
    if not source.exists():
        sys.exit(f"[error] '{source}' not found")
    return [source]


def _read_sentences(args: argparse.Namespace) -> List[str]:
    sentences: List[str] = []
    for f in _text_files(Path(args.source)):
        text = f.read_text(encoding="utf-8")
        if args.parse:
            sentences += parse_text(
                text,
                min_words=args.min_words,
                remove_digits=not args.keep_digits,
                stopwords=STOPWORDS[args.lang] if args.remove_stopwords else None,
            )
        else:
            sentences += [line for line in text.splitlines() if line.strip()]
    return sentences


def _load(map_dir: str) -> LogPrime:
    try:
        return LogPrime.load(map_dir)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[error] {e}")


def cmd_build(args: argparse.Namespace) -> None:
    sentences = _read_sentences(args)
    lp = _load(args.output) if args.add else LogPrime()
    before = len(lp)
    t0 = time.perf_counter()
    lp.add(sentences)
    lp.save(args.output)
    print(
        f"[ok] {len(sentences)} sentences → {args.output}/  "
        f"({len(lp)} words, +{len(lp) - before} new, {time.perf_counter() - t0:.1f}s)"
    )


def cmd_info(args: argparse.Namespace) -> None:
    lp = _load(args.map_dir)
    print(f"map      : {args.map_dir}/")
    print(f"words    : {len(lp)}")
    print(f"entries  : {lp.matrix.nnz}")
    if args.word:
        word = args.word.lower()
        if word not in lp.vocabulary:
            sys.exit(f"[not found] '{word}' is not in the map.")
        print(f"\n'{word}' (prime {lp.vocabulary[word]}), top 20 components:")
        top = sorted(lp.embedding(word).items(), key=lambda kv: -kv[1])[:20]
        for other, val in top:
            print(f"  {other:20s} {val:8.4f}")


def cmd_decode(args: argparse.Namespace) -> None:
    lp = _load(args.map_dir)
    tokens = args.sentence.lower().split()
    unknown = [t for t in tokens if t not in lp.vocabulary]
    if unknown:
        print(f"[warn] not in map, ignored: {' '.join(unknown)}")
    known = [t for t in tokens if t in lp.vocabulary]
    vec = lp.vector(args.sentence)
    t0 = time.perf_counter()
    words = lp.decode(vec, n_words=len(known))
    elapsed = time.perf_counter() - t0
    print(f"input   : {' '.join(known)}")
    print(f"decoded : {' '.join(words)}  ({elapsed * 1000:.0f} ms)")
    if lp.verify(words, vec):
        print("[ok] verified: the decoded words reproduce the vector exactly")
    else:
        print("[fail] not verified: the decoded words do not reproduce the vector")


def cmd_code(args: argparse.Namespace) -> None:
    lp = _load(args.map_dir)
    codec = lp.codec(d=args.d, k=args.k, orders=range(2, args.max_n + 1), positions=args.positions)
    tokens = args.sentence.lower().split()
    unknown = [t for t in tokens if t not in lp.vocabulary]
    if unknown:
        print(f"[warn] not in map, ignored: {' '.join(unknown)}")
    known = [t for t in tokens if t in lp.vocabulary]
    code = codec.encode(known)
    t0 = time.perf_counter()
    result = codec.decode(code)
    elapsed = time.perf_counter() - t0
    print(f"code    : {code.shape[0]} × {code.shape[1]} integers, {int((code > 0).sum())} non-zero")
    print(f"input   : {' '.join(known)}")
    print(f"decoded : {result.text}  ({elapsed * 1000:.0f} ms)")
    messages = {
        "exact": "[ok] exact: the only word order that fits, verified against the code",
        "ambiguous": "[ambiguous] several word orders reproduce the code exactly; another one:",
        "bag": "[bag] the words are verified, their order could not be recovered",
        "fail": "[fail] the code could not be decoded",
    }
    print(messages[result.status])
    for alt in result.alternatives:
        print(f"          {' '.join(alt)}")


def cmd_semantic(args: argparse.Namespace) -> None:
    path = Path(args.map_dir) / "semantic.npz"
    sums = Path(args.map_dir) / "sums.npz"
    if path.exists() and not args.rebuild and path.stat().st_mtime >= sums.stat().st_mtime:
        sem = Semantic.load(path)
    else:
        lp = _load(args.map_dir)
        t0 = time.perf_counter()
        sem = lp.semantic(dim=args.dim)
        sem.save(path)
        print(f"[ok] word vectors ({sem.vectors.shape[1]} dims) → {path}  ({time.perf_counter() - t0:.0f}s)")
    for word in args.word or []:
        word = word.lower()
        if word not in sem.index:
            print(f"[not found] '{word}'")
            continue
        near = ", ".join(w for w, _ in sem.neighbours(word, n=args.n, min_count=args.min_count))
        print(f"{word:15s} {near}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="logprime", description="Word co-occurrence map, exact sentence codes, word vectors.")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build a map from text, or add text to one.")
    b.add_argument("source", help="A .txt file or a directory of .txt files.")
    b.add_argument("-o", "--output", required=True, metavar="DIR", help="Map directory.")
    b.add_argument("-a", "--add", action="store_true", help="Add to the existing map in DIR.")
    b.add_argument("-p", "--parse", action="store_true",
                   help="Split raw text into clean sentences. Without it: one sentence per line.")
    b.add_argument("--lang", choices=sorted(STOPWORDS), default="it", help="Stopword language (default: it).")
    b.add_argument("--remove-stopwords", action="store_true", help="Drop stopwords (with --parse).")
    b.add_argument("--min-words", type=int, default=2, metavar="N",
                   help="Drop sentences shorter than N words (with --parse, default: 2).")
    b.add_argument("--keep-digits", action="store_true", help="Keep numbers (with --parse).")
    b.set_defaults(func=cmd_build)

    i = sub.add_parser("info", help="Show a map's size, or one word's embedding.")
    i.add_argument("map_dir")
    i.add_argument("-w", "--word", help="Show the top components of this word.")
    i.set_defaults(func=cmd_info)

    d = sub.add_parser("decode", help="Encode a sentence and decode it back.")
    d.add_argument("map_dir")
    d.add_argument("sentence")
    d.set_defaults(func=cmd_decode)

    c = sub.add_parser("code", help="Encode a sentence as a fixed-size code and decode it back, order included.")
    c.add_argument("map_dir")
    c.add_argument("sentence")
    c.add_argument("--d", type=int, default=1024, help="Cells per channel (default: 1024).")
    c.add_argument("--k", type=int, default=8, help="Cells per word (default: 8).")
    c.add_argument("--max-n", type=int, default=5, metavar="N",
                   help="Longest n-gram channel used for the order (default: 5; 1 = words only).")
    c.add_argument("--positions", type=int, default=8, metavar="B",
                   help="Slot size of the position channel (default: 8; 0 = none).")
    c.set_defaults(func=cmd_code)

    s = sub.add_parser("semantic", help="Word vectors from the map (cached in DIR/semantic.npz).")
    s.add_argument("map_dir")
    s.add_argument("-w", "--word", action="append", help="Show the nearest words (repeatable).")
    s.add_argument("--dim", type=int, default=300, help="Vector size (default: 300).")
    s.add_argument("-n", type=int, default=10, help="Neighbours to show (default: 10).")
    s.add_argument("--min-count", type=int, default=5, help="Ignore rarer neighbours (default: 5).")
    s.add_argument("--rebuild", action="store_true", help="Recompute even if cached.")
    s.set_defaults(func=cmd_semantic)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
