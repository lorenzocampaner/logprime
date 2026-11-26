# LogPrime

Exact, verifiable text encodings. LogPrime has three parts:

1. **The map** (`LogPrime`). Each word gets a prime, and the map stores exact
   integer co-occurrence sums. A sentence vector decodes back into its exact
   words.
2. **The codec** (`Codec`). Every sentence becomes a fixed-size integer code
   (by default 6 × 1024) that decodes back to the same sentence, word order
   included. Every answer is checked against the code. With word vectors, one
   more row carries the sentence's meaning, so two codes can be compared
   without decoding them.
3. **Word vectors** (`Semantic`). Dense vectors for word similarity, computed
   from the counts the map already holds, without reading the text again.

I'm not a professional software engineer. This repository started from an
original handwritten idea of mine and was developed with AI assistance. :)

No training and no gradient descent: every value has an explicit formula.

---

## When to use it

LogPrime is at its best as a **transparent, reversible sentence encoder**:

- **Exact, verifiable inversion.** The words of a sentence, and with the codec
  their order, are recovered from the vector alone, with no header, metadata
  or side channel. Every decode is re-encoded and compared, so an answer is
  either certified or reported as not certified. It is never silently wrong.
- **Fixed size.** A codec code has the same shape for every sentence and every
  vocabulary size. Codes can be added: the sum of several sentences' codes
  decodes to all their words.
- **Decodable arithmetic.** `vec(a) − vec(word) + vec(other)` decodes back to
  words, so you can edit and inspect sentence vectors. This works when the
  result is still a sum of words (you remove only words that are there).
- **Meaning and exact text in one vector.** With the meaning row, the same
  code gives a similarity by meaning (a cosine, no decoding) and the exact
  sentence back. Usual embeddings give only the first; storing the text next
  to them gives both, but nothing ties the two together.
- **Fully interpretable.** Every dimension and every value has a formula.
- **Incrementally updatable.** New text can be added to a map, with exactly the
  same result as a rebuild. Codec codes do not change when new words are
  appended to the vocabulary.

It is **not** a compressor: a code is much larger than the text it encodes. And
it is a modest semantic model: the word vectors are the classic count-based
kind, and on a corpus of one million words they are far from modern embeddings
(see [Results](#results)).

---

## The idea

### 1. The map

Each word gets a distinct prime, in order of first appearance:
`io → 2`, `mangio → 3`, `la → 5`, `mela → 7`, …

For every occurrence of a word wᵢ (prime pᵢ), each word wⱼ that **follows** it
in the sentence adds its prime pⱼ to the cell (wᵢ, wⱼ). The word also adds its
own prime to its own cell (wᵢ, wᵢ). Over the whole corpus:

```
S[wᵢ][wⱼ] = Σ pⱼ                    exact integer sum
M[wᵢ][wⱼ] = ln(pᵢ · S[wᵢ][wⱼ])      embedding value
```

- The row `M[wᵢ]` is the **embedding** of wᵢ: its context, one dimension per known word.
- A **sentence vector** is the sum of the rows of its words.
- **Decoding** recovers the words of a sentence (as a multiset) from its vector alone.

Example: after `io mangio la mela`, the row for `io` is
`[ln(2·2), ln(2·3), ln(2·5), ln(2·7)]` and the row for `mela` is `[ln(7·7)]`.

Only the integer sums `S` are stored. Adding new text to a map therefore gives
exactly the same result as rebuilding it from scratch. And because
`S[wᵢ][wⱼ] / pⱼ` is the number of times wⱼ followed wᵢ, the map also holds the
co-occurrence counts that the word vectors are built from.

### 2. The codec

A map vector has one dimension per word, so it grows with the corpus and
decoding gets slower. The codec keeps the idea, a sum that can be taken apart
again, but in a fixed size:

- **Word channel.** The d cells are split into k blocks. Each word lights one
  cell per block, and the value in those cells is the word's prime: the
  same prime it has in the map. Which cells it lights comes from a hash of the
  word. A sentence is the sum of its words' codes.
- **n-gram channels.** One more row per n = 2, 3, 4, 5: the sum of a hashed
  code for every run of n adjacent words of `<s> w₁ … w_L </s>`. This is
  the original idea of encoding a word's context, extended from pairs to
  longer runs.
- **Position channel.** One more row: the sum of a hashed code for every pair
  (word, its position ÷ 8). It is coarse on purpose. It only has to tell apart
  the rare sentences whose n-grams are identical (see
  [Why n-grams alone are not enough](#why-n-grams-alone-are-not-enough)).
- **Meaning channel** (optional, with word vectors). One last row: the sum of
  the words' vectors, as integers. It is what makes two codes comparable by
  meaning; decoding does not need it.

### 3. Word vectors

The map's counts, made symmetric, are sentence-level co-occurrence counts.
`Semantic` turns them into positive PMI, then reduces them with a truncated
SVD to 300 dimensions: the classic count-based model
([Levy, Goldberg & Dagan, 2015](https://aclanthology.org/Q15-1016/)).

---

## Quick start

```bash
git clone https://github.com/lorenzocampaner/logprime
cd logprime
pip install -e .
```

Any plain-text corpus works. Public-domain books from
[Project Gutenberg](https://www.gutenberg.org) or
[Liber Liber](https://www.liberliber.it) are a good start.

```bash
logprime build texts/ -o map/ --parse                # raw .txt files → map
logprime code map/ "la guerra continua senza fine"   # fixed-size code, decoded with its order
logprime decode map/ "la guerra continua senza fine" # map vector, decoded as a bag of words
logprime semantic map/ -w guerra -w notte            # nearest words
```

```python
from logprime import LogPrime
from logprime.parser import parse_dir

lp = LogPrime().add(parse_dir("texts/"))
lp.save("map/")

codec = lp.codec()
code = codec.encode("la guerra continua senza fine")   # int64 array, shape (6, 1024)
codec.decode(code)
# Decoded(status='exact', words=['la', 'guerra', 'continua', 'senza', 'fine'], alternatives=[])

sem = lp.semantic()
sem.neighbours("notte", n=5)   # [('giorno', …), ('dormiva', …), ('accorti', …), ('camera', …), ('sera', …)]
```

---

## Python API

### The map

```python
lp = LogPrime()
lp.add(sentences)               # whitespace-separated sentences; call as often as you like
lp.save("map/")
lp = LogPrime.load("map/")

lp.vocabulary                   # {word: prime}
lp.words                        # words in prime order = row/column labels
lp.matrix                       # scipy.sparse CSR matrix M, shape (n_words, n_words)
lp.embedding("guerra")          # {context_word: value}

lp.vector("la guerra continua")                           # numpy array; unknown words are ignored
lp.similarity("la guerra continua", "la guerra finisce")  # cosine, projected on the words compared

lp.decode(vec)                  # multiset of words, repeats included
lp.decode(vec, n_words=5)       # faster when the number of words is known (a hint, not a limit)
lp.verify(words, vec)           # True only if the words' rows sum exactly to vec

# vector arithmetic: works on any combination that is itself a sum of words
lp.decode(lp.vector("il gatto mangia il topo") - lp.vector("topo") + lp.vector("pesce"))
# → ['il', 'il', 'pesce', 'gatto', 'mangia']  (a multiset: order is lost)
```

### The codec

```python
from logprime import Codec

codec = lp.codec()                       # same as Codec(lp.words)
codec = Codec(words, d=1024, k=8, orders=(2, 3, 4, 5), positions=8, seed=0)

code = codec.encode("il gatto mangia il topo")   # or a list of tokens; unknown words raise ValueError
r = codec.decode(code)
r.status        # "exact" | "ambiguous" | "bag" | "fail"
r.words         # the sentence (for "bag": the words in no particular order)
r.text          # " ".join(r.words)
r.alternatives  # for "ambiguous": other orders that fit the code exactly

codec.verify("il gatto mangia il topo", code)    # True only if it encodes to exactly this code
codec.decode(codec.encode("il gatto") + codec.encode("la nonna")).status   # "bag": all four words

codec.save("codec/")
codec = Codec.load("codec/")

codec = lp.codec(semantic=lp.semantic())         # adds the meaning row (7 × 1024)
codec.similarity("la guerra continua", "la battaglia infuria")   # cosine of meanings, from the codes
codec.similarity(code_a + code_b, code_c)        # codes, sums of codes: the meaning row adds up too
```

**The meaning row.** With `semantic=`, every code gets one more row: the sum
of its words' unit vectors, ×1000 and rounded to integers, in the first `dim`
cells. It is linear like the other rows, so the meaning of a sum of codes is
the meaning of all their words. `similarity` reads it straight from two codes.
Decoding ignores it, and verification checks it like every other row: a code
whose meaning does not match its words is not certified. Any vectors with
`.index` and `.vectors` work, so better vectors than `Semantic`'s (trained on
more text) give better meaning with no other change.

| Status | Meaning |
|---|---|
| `exact` | the order was recovered, it is the only one that fits, and it reproduces the code exactly |
| `ambiguous` | several orders reproduce the code exactly. With the default channels this never happened on our corpus; it can with `positions=0` (see [Why n-grams alone are not enough](#why-n-grams-alone-are-not-enough)) |
| `bag` | the words are certified, the order is not: no n-gram channels (`orders=()`), or a sum of several sentences |
| `fail` | nothing could be certified, e.g. a sentence too long for `d` |

**Choosing d and k.** `d = 1024, k = 8` decodes every sentence up to 80 words
on a 58k-word vocabulary. Use `d = 2048` or more for longer sentences or larger
vocabularies (see [Results](#results)). `orders=()` keeps only the word
channel: a single row, and the bag of words. `positions=0` drops the position
channel.

### Word vectors

```python
from logprime import Semantic

sem = lp.semantic(dim=300)          # same as Semantic.from_map(lp, dim=300)
sem.vector("guerra")                # unit-length float32 vector; a text gives the mean
sem.similarity("notte", "giorno")   # cosine
sem.neighbours("notte", n=10, min_count=5)   # [(word, cosine), …], rare words left out
sem.save("semantic.npz")
sem = Semantic.load("semantic.npz")
```

On the 58k-word map, the SVD takes a few minutes. `logprime semantic` caches the
result in `map/semantic.npz`.

### Text parser

```python
from logprime.parser import parse_text, parse_file, parse_dir, STOPWORDS_IT, STOPWORDS_EN

parse_text("L'uomo mangiò la mela. Poi bevve.")
# → ['uomo mangiò la mela', 'poi bevve']
```

The parser splits sentences on `. ! ? ;` and on blank lines. Single line breaks
count as spaces, so hard-wrapped books work. It lowercases the text, keeps the
word after an apostrophe (`dell'aria → aria`) and strips punctuation and
digits. Optionally it removes stopwords and drops sentences shorter than
`min_words`.

---

## CLI

```bash
logprime build SOURCE -o MAP_DIR [--add] [--parse] [--lang it|en]
               [--remove-stopwords] [--min-words N] [--keep-digits]
logprime info MAP_DIR [--word WORD]
logprime decode MAP_DIR "a sentence"                    # map vector → bag of words
logprime code MAP_DIR "a sentence" [--d D] [--k K] [--max-n N]   # codec round trip
logprime semantic MAP_DIR [-w WORD ...] [--dim D] [-n N] [--min-count C] [--rebuild]
```

`SOURCE` is a `.txt` file or a directory of `.txt` files. Without `--parse`,
each non-empty line is one sentence. `--add` adds the text to the map already
in `MAP_DIR`.

A map directory contains `vocabulary.json` (word → prime) and `sums.npz` (the
sparse integer matrix `S`), plus `semantic.npz` once `logprime semantic` has
run. The word vectors are recomputed when the map is newer.

---

## How decoding works

### The map

A sentence vector is `vec = xᵀ M`, where `x` counts how many times each word
occurs. Decoding finds `x`:

1. **Candidates.** All values in `M` are positive. A word can therefore only
   be in the sentence if every column of its row is non-zero in `vec`.
2. **Greedy selection with over-selection.** Orthogonal Matching Pursuit picks
   the candidate that best explains what is left, refits all picks by least
   squares, and repeats. It may go on for up to 3× the expected number of
   words. Wrong picks end up with a coefficient near 0, true words near their
   count (1, 2, …).
3. **Rounding.** Rounding the coefficients gives the counts.
4. **Verification.** `verify` rebuilds the vector from the decoded words and
   compares. If it matches, the answer is exact: in every map we tested, `M`
   was invertible, and then only one multiset gives this vector. If the first
   attempt does not verify, `decode` searches again without the bound.

The inner products the search needs are cached between calls, so frequent
words like `il` or `di` are computed only once per session.

### The codec

1. **Words, read off the cells.** A cell that holds a single word holds
   n·p: n occurrences of the word with prime p. Factoring the value gives the
   word and its count directly. A reading is accepted when every other cell
   of that word holds at least n·p; the word is then subtracted, which leaves
   more cells with a single word, and so on.
   This is *peeling*, the decoder of invertible Bloom lookup tables. On real
   sentences it decodes almost everything by itself.
2. **Words, the rest.** Whatever peeling leaves is solved by least squares:
   a word is a candidate only if all its k cells are non-zero, and the fit on
   the candidates, rounded, gives the counts (sparse recovery, as in
   compressed sensing). When candidate codes overlap, the fit can have a whole
   line of real solutions; the decoder then searches that line for the
   integer point.
3. **Bigrams.** The only possible bigrams are pairs of decoded words, plus
   `<s>` and `</s>`. The bigram row is solved the same way, among those pairs.
4. **Longer n-grams.** The only possible trigrams are pairs of bigrams that
   overlap (`a b` + `b c` → `a b c`), and so on up to 5-grams.
5. **Positions.** The position row is solved among the pairs (decoded word,
   slot), which says how many times each word sits in each block of 8
   positions.
6. **Order.** The sentence is a path from `<s>` to `</s>` that uses every
   5-gram exactly once (an Eulerian path), and puts each word only in a slot
   the position row allows. The search stops at the second path it finds.
7. **Verification.** Every path found is re-encoded and compared with the whole
   code. One path means `exact`, several mean `ambiguous`.

### Why n-grams alone are not enough

The n-gram rows record which runs of n words occur, not where. Take a sentence
`A R B R C R D`, where `R` is a stretch of at least n − 1 words that occurs
three times. Then `A R C R B R D` has exactly the same n-grams, so the two
sentences have the same code, and no decoder can tell them apart. This is the
same limit as rebuilding a DNA sequence from its k-mers. Longer n-grams only
help if they are longer than the repeat. On real text this happens in about 1
sentence in 5,000, all of them long ones that repeat a refrain several
times.

The position channel removes the ambiguity. Swapping `B` and `C` moves their
words by many positions, so they land in other slots, and the position row no
longer matches. The slots are coarse (8 positions), so the row is cheap to
decode. On the whole corpus no sentence stayed ambiguous.

---

## Results

Benchmarks are in `benchmarks/`, and each script explains how to run it. The
corpus is 11 Italian texts (fairy tales, Galileo, 19th-century travel and
memoirs, novels): 51k sentences, 1.05M tokens, 58k words. Times are on a 2019
laptop (Intel Core i5, 2.4 GHz).

### The map

**Exact decoding** (`decode.py`). One map per book, each tested with 100
sentences and 100 random bags of 1–60 vocabulary words, repeats included:
**2,053 of 2,053** cases decoded exactly. There were no wrong answers that
passed verification. On the three smallest books, a sparse LU factorization
also confirmed that `M` is invertible.

**Speed**, all 11 texts in one map (6M entries). The map builds in 3–4 s and is
26 MB on disk.

| Sentence length | 1–5 | 6–10 | 11–20 | 21–40 | 41–80 |
|---|--:|--:|--:|--:|--:|
| Exact | 100% | 100% | 100% | 100% | 100% |
| ms per sentence | 47 | 63 | 98 | 158 | 305 |

On a single book (10k words), a sentence takes 3–20 ms.

### The codec

**Words and order** (`codec.py`). Every sentence of the corpus (51,048), with
the default codec: d = 1024, k = 8, n-grams up to 5, positions in slots of 8:

| Sentence length | 1–10 | 11–20 | 21–40 | 41–80 | 81+ |
|---|--:|--:|--:|--:|--:|
| Sentences | 15,932 | 15,647 | 13,923 | 4,968 | 578 |
| **Exact, order included** | **100%** | **100%** | **100%** | **100%** | 96.2% |
| Ambiguous order | 0 | 0 | 0 | 0 | 0 |
| Words only, order not recovered | 0 | 0 | 0 | 0 | 3.8% |
| Not decoded | 0 | 0 | 0 | 0 | 0 |
| ms per sentence | 5 | 6 | 9 | 18 | 43 |

Every sentence's words are recovered. For 4% of the very long ones the order
is not: their n-gram channels are too full for d = 1024. With **d = 2048
every sentence decodes exactly**: all 578 of 81+ words, and 600 per shorter
length tested, at 5–38 ms per sentence.

No certified answer was ever wrong. The sentences that are not decoded are
reported as such, never answered wrongly.

**How long can a sentence be?** Words only, all 11 texts (58k words), 300
sentences per length:

| d | k | 11–20 | 21–40 | 41–80 | 81+ | ms |
|--:|--:|--:|--:|--:|--:|--:|
| 256 | 8 | 100% | 100% | 100% | 99.0% | 1.4 |
| 512 | 8 | 100% | 100% | 100% | 100% | 1.3 |
| 1024 | 8 | 100% | 100% | 100% | 100% | 1.2 |
| 4096 | 8 | 100% | 100% | 100% | 100% | 1.5 |

With least squares alone, d = 256 stopped at about 20 words and d = 512 at
about 40. Random bags of vocabulary words (repeats possible), 20 per size,
share decoded exactly (`codec.py --bags`):

| d | k | 300 | 400 | 500 | 600 | 700 | 800 | 1000 |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1024 | 8 | 100% | 100% | 75% | 0% | 0% | 0% | 0% |
| 1024 | 4 | 100% | 100% | 100% | 100% | 95% | 15% | 0% |
| 1023 | 3 | 100% | 100% | 100% | 100% | 100% | 95% | 0% |
| 4096 | 8 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

With least squares alone, the largest bags that always decoded were 50 words
for d = 1024 and 400 for d = 4096. The limit is sharp: just below it
everything decodes, just above it nothing does. With k = 3 it is about one
word per 1.3 cells, close to the 1.22 that the theory of invertible Bloom
lookup tables predicts. Small k is best for bags of words
(`orders=()`); with the order channels, k = 8 stays the better default,
because the n-gram channels need the extra cells (81+ words: 96.7% exact with
k = 8, 82% with k = 3).

**Which channels does the order need?** Every sentence of the corpus, d = 1024,
share decoded in the right order, without and with the position channel:

| n-grams up to | 21–40 | 41–80 | 81+ | 21–40, positions | 41–80, positions | 81+, positions |
|---|--:|--:|--:|--:|--:|--:|
| 2 | 48.2% | 3.7% | 0% | 94.4% | 87.5% | 70.4% |
| 3 | 98.8% | 91.9% | 56.2% | 99.9% | 99.7% | 94.8% |
| 4 | 99.9% | 99.1% | 90.7% | **100%** | **100%** | 96.0% |
| 5 | **100%** | 99.9% | 95.2% | **100%** | **100%** | 96.2% |
| 6 | **100%** | **100%** | 95.5% | **100%** | **100%** | 96.2% |

Sentences of up to 20 words are 90–100% exact with bigrams alone without
positions, and 98–100% with them. In the 81+ column, 3.8% is the capacity
limit of d = 1024 (words right, order not recovered) in every row; the rest
is ambiguity. Without positions, 4 of the 578 longest sentences stay
ambiguous even with 6-grams: their repeats are longer than 6 words. With
positions, none does from 5-grams on, which is the default.

### Word similarity

Human similarity judgements: the Italian SimLex-999 and WordSim-353 of
[Leviant & Reichart (2015)](https://github.com/iraleviant/eval-multilingual-simlex).
Every method is built from the same 11 texts. Spearman ρ between human scores
and model similarity (`*` = p < 0.05), from `similarity.py` and `semantics.py`:

| Method                                       | SimLex-999 | WS-353  | WS-353 sim | WS-353 rel |
|----------------------------------------------|-----------:|--------:|-----------:|-----------:|
| **`Semantic`** (PPMI + SVD 300, map counts)  | 0.040      | 0.195*  | 0.256*     | 0.042      |
| PPMI, map counts                             | −0.002     | 0.186*  | 0.255*     | **0.145**  |
| map `similarity()`                           | −0.013     | 0.223*  | 0.323*     | 0.105      |
| same projection, presence only (no primes)   | −0.015     | 0.215*  | 0.312*     | 0.105      |
| map, cosine of full rows                     | −0.040     | 0.087   | 0.155      | 0.090      |
| PPMI, ±5 window (reads the text again)       | 0.035      | 0.162*  | 0.325*     | 0.072      |
| PPMI + SVD 300, ±5 window                    | 0.008      | **0.246*** | **0.393*** | 0.143  |
| word2vec skip-gram 300                       | **0.094*** | 0.036   | 0.136      | −0.062     |

Pairs covered: 857/999, 242/350, 145/201, 180/252.

- **`Semantic` comes for free.** It needs only the map, and it scores close to
  the same model built from a ±5-word window. That model is better on
  similarity (WS-353 sim 0.393), because nearby words say more than the whole
  sentence.
- **The map's `similarity()` only sees direct co-occurrence.** For two words it
  compares `[M[a][a], M[a][b]]` with `[M[b][a], M[b][b]]`. If two words never
  appear in the same sentence, their similarity is 0, which holds for 192 of the
  242 WS-353 pairs. `Semantic` also links words that never meet, through shared
  contexts.
- **Nearest neighbours** of `Semantic` are sensible for frequent words
  (`partito → membri, membro, rivoluzione`, `notte → giorno, dormiva, camera,
  sera`, `occhi → viso, volto, labbra, fronte`) and noisy for rare ones.
- **The corpus is small.** One million tokens is far too little for word2vec,
  which is why it is weak here, and all the numbers are far below published
  results on large corpora. The ranking between the count-based methods is the
  meaningful part.

---

## What the primes do (and don't)

The original idea used primes for two reasons: **context** and
**invertibility**. Measured:

- **Invertibility does not need the primes.** It comes from the structure:
  each row marks which words follow wᵢ, all values are positive, and there is
  a self cell on the diagonal. The self cell matters most. Without it, a word
  that never has a follower would have an empty row and could not be decoded.
  With the same structure, almost any positive values decode exactly
  (`cell_values.py`, one book, 200 sentences and 200 random bags):

  | Value of cell (wᵢ, wⱼ)         | Sentences | Random bags |
  |--------------------------------|----------:|------------:|
  | `ln(pᵢ · Σpⱼ)` (LogPrime)      | 100%      | 100%        |
  | `Σ ln(pᵢ · pⱼ)` (product)      | 100%      | 100%        |
  | `ln(1 + count)` (no primes)    | 100%      | 100%        |
  | random in [1, 2]               | 100%      | 100%        |
  | `1` (presence only)            | 99.5%     | 97.5%       |

  The codec makes the same point: with random integers instead of primes as
  word values, it decodes exactly the same sentences.
- **Context comes from co-occurrence, not from the primes.** The map's
  similarity scores are nearly the same when every value is replaced with `1`.
- **What the primes do give:** a deterministic, never-repeating weight for
  every word. Through ln, products become sums, so vectors can be added and
  subtracted. They are a valid, canonical choice of weights, not the cause of
  the properties above. And since `S[wᵢ][wⱼ] / pⱼ` is a count, the map keeps
  the exact counts that `Semantic` is built from.
- **Integer solutions do not need them either.** Each word's value in the
  codec is its prime, the same in all its cells. When candidate codes overlap
  heavily, least squares can have a line or plane of real solutions, and
  rounding the point it picks is wrong. The decoder then searches that line
  for the non-negative integer point. On codes whose cells were chosen by
  meaning (an earlier design, see below; 40 seeds, 640 sentences) this turns
  29 failures into 0, and it does the
  same with random, consecutive or even values (19, 27, 27 failures before,
  0 after). The search does the work; any distinct values will do.
- **Peeling does not need them either, but it reads them.** The codec decoder
  reads each word off a cell by factoring its value n·p. Random distinct
  values do as well (random bags on 50k words, d = 1024, k = 8, 10 per size:
  primes 10, 8, 1 of 10 at 450, 500, 550 words; random values 10, 8, 2).
  Consecutive values do worse (5, 5, 0): when every integer is some word's
  value, a sum of several words is easily misread as one, and peeling stalls.
  What the primes add is that a cell's value is legible: n·p says, by
  itself, which word and how many times.
- **Where unique factorization would matter.** A sentence as the product of
  its words' primes is a single integer that determines the multiset exactly,
  and divisibility gives exact containment tests (`N % p == 0`) and shared
  words (`gcd`). It carries the same information as a list of word ids and no
  meaning, and it is lost as soon as it is turned into floating point.

## What does not work (tested)

- **Choosing cells by meaning.** An earlier `Codec(semantic=sem)` chose each
  word's cells with a winner-take-all hash of its vector, so that similar
  words would share cells. Decoding stayed exact, but the similarity read off
  shared cells was weak and unstable: with five seeds, WS-353 ranged from
  −0.01 to 0.22. The meaning row replaced it: it matches `Semantic` exactly
  (WS-353 0.196 against 0.195; SimLex 0.040 for both), does not depend on the
  seed, and leaves decoding untouched (`code_semantics.py`).
- **Semantic analogies.** We tested 7 cases with cosine similarity and got 1
  right (`padre − uomo + donna → madre`, ranked 5th). That is probably a
  coincidence.
- **Compression.** Neither the map vectors nor the codes are smaller than the
  word ids they encode. A random projection of a map vector to 10% of its
  dimensions still decodes exactly, but so does the same projection of the
  plain word-count vector.

## Open questions

- **Ambiguity with positions.** No sentence of our corpus stayed ambiguous,
  but it is not impossible. Two swapped parts would have to contain the same
  words in the same slots of 8. `positions=1` rules it out entirely, at a
  higher decoding cost.
- **Proof of invertibility.** `M` was invertible on every map we checked, but
  there is no proof that it always is. For the codec, compressed sensing gives
  the theory for random codes. A bound for these particular codes is still to
  be written.
- **Better meaning from the same text.** The meaning row is only as good as
  the word vectors, and 1M tokens is little. Pooling the counts of a word's
  inflected forms (gatto, gatti, gatta) before PPMI and SVD helped in a first
  test: WS-353 sim 0.256 → 0.391, SimLex 0.040 → 0.084 (now significant). It is
  not in the library: the stemming used (dropping final vowels) is
  Italian-only and crude.

---

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests use a small hand-written corpus and need no external data.

## License

MIT
