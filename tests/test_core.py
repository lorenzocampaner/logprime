import math

import numpy as np
import pytest

from logprime import LogPrime


# --- encoding --------------------------------------------------------------

def test_primes_assigned_in_order_of_appearance():
    lp = LogPrime().add(["io mangio la mela", "io compro"])
    assert lp.vocabulary == {"io": 2, "mangio": 3, "la": 5, "mela": 7, "compro": 11}
    assert lp.words == ["io", "mangio", "la", "mela", "compro"]


def test_single_sentence_values():
    lp = LogPrime().add(["io mangio la mela"])
    # forward-only: each word sees only the words after it, plus itself
    assert lp.embedding("io") == pytest.approx({
        "io": math.log(2 * 2),
        "mangio": math.log(2 * 3),
        "la": math.log(2 * 5),
        "mela": math.log(2 * 7),
    })
    assert lp.embedding("mela") == pytest.approx({"mela": math.log(7 * 7)})


def test_values_accumulate_primes_then_take_log():
    lp = LogPrime().add(["io mangio", "io compro", "io mangio"])
    # io(2) is followed by mangio(3) twice: ln(2 * (3 + 3))
    e = lp.embedding("io")
    assert e["mangio"] == pytest.approx(math.log(2 * 6))
    assert e["compro"] == pytest.approx(math.log(2 * 5))
    assert e["io"] == pytest.approx(math.log(2 * (2 + 2 + 2)))


def test_repeated_word_in_sentence():
    lp = LogPrime().add(["a b a"])
    # first "a" sees b and a; second "a" sees nothing after it
    assert lp.embedding("a") == pytest.approx({"a": math.log(2 * (2 + 2 + 2)), "b": math.log(2 * 3)})


def test_input_is_lowercased():
    lp = LogPrime().add(["Il Gatto"])
    assert set(lp.vocabulary) == {"il", "gatto"}


def test_incremental_add_equals_full_build(corpus):
    full = LogPrime().add(corpus)
    inc = LogPrime().add(corpus[:5]).add(corpus[5:11]).add(corpus[11:])
    assert full.vocabulary == inc.vocabulary
    assert (full.matrix != inc.matrix).nnz == 0


# --- vectors ---------------------------------------------------------------

def test_vector_is_sum_of_rows(lp):
    v = lp.vector("gatto topo")
    expected = lp.matrix.getrow(lp.words.index("gatto")) + lp.matrix.getrow(lp.words.index("topo"))
    assert np.allclose(v, expected.toarray().ravel())


def test_vector_ignores_unknown_words(lp):
    assert np.allclose(lp.vector("gatto xyzzy"), lp.vector("gatto"))
    assert not lp.vector("xyzzy").any()


def test_similarity_projects_on_the_words_compared():
    lp = LogPrime().add(["a b", "b c"])
    # a=2, b=3, c=5. Columns [a, b]: a -> [ln(2·2), ln(2·3)], b -> [0, ln(3·6)]
    va = np.array([math.log(4), math.log(6)])
    vb = np.array([0.0, math.log(18)])
    expected = va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))
    assert lp.similarity("a", "b") == pytest.approx(expected)
    assert lp.similarity("a", "c") == 0.0  # never in the same sentence


def test_similarity(lp):
    assert lp.similarity("il gatto", "il gatto") == pytest.approx(1.0)
    assert -1.0 <= lp.similarity("il gatto", "la pioggia") <= 1.0
    assert lp.similarity("xyzzy", "il gatto") == 0.0


# --- decoding --------------------------------------------------------------

def test_decode_every_corpus_sentence_exactly(lp, corpus):
    for s in corpus:
        vec = lp.vector(s)
        words = lp.decode(vec)
        assert sorted(words) == sorted(s.split()), s
        assert lp.verify(words, vec)


def test_decode_random_bags_with_repeats(lp):
    rng = np.random.default_rng(0)
    for _ in range(50):
        bag = list(rng.choice(lp.words, size=rng.integers(1, 30)))
        vec = lp.vector(" ".join(bag))
        words = lp.decode(vec, n_words=len(bag))
        assert sorted(words) == sorted(bag)
        assert lp.verify(words, vec)


def test_decode_repeated_word(lp):
    assert sorted(lp.decode(lp.vector("il il il gatto"))) == ["gatto", "il", "il", "il"]


def test_decode_arithmetic(lp):
    vec = lp.vector("il gatto mangia il topo") - lp.vector("topo") + lp.vector("pesce")
    assert sorted(lp.decode(vec)) == ["gatto", "il", "il", "mangia", "pesce"]


def test_decode_empty_vector(lp):
    assert lp.decode(lp.vector("xyzzy")) == []


def test_verify_rejects_wrong_words(lp):
    vec = lp.vector("il gatto mangia il topo")
    assert not lp.verify(["il", "gatto", "mangia", "topo"], vec)
    assert not lp.verify(["il", "cane", "mangia", "il", "topo"], vec)


def test_decode_reports_unrepresentable_vector(lp):
    # "gatto" minus "cane" is not a sum of words: decoding cannot verify
    vec = lp.vector("gatto") - lp.vector("cane")
    assert not lp.verify(lp.decode(vec), vec)


# --- persistence -----------------------------------------------------------

def test_save_load_roundtrip(tmp_path, lp):
    lp.save(tmp_path / "map")
    loaded = LogPrime.load(tmp_path / "map")
    assert loaded.vocabulary == lp.vocabulary
    assert (loaded.matrix != lp.matrix).nnz == 0


def test_loaded_map_continues_prime_sequence(tmp_path):
    LogPrime().add(["io mangio"]).save(tmp_path)
    lp = LogPrime.load(tmp_path).add(["io corro"])
    assert lp.vocabulary["corro"] == 5


def test_load_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LogPrime.load(tmp_path / "nope")
