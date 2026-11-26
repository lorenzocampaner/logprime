import random
from collections import Counter

import numpy as np
import pytest

from logprime import Codec, LogPrime


@pytest.fixture
def codec(lp):
    return lp.codec()


def test_code_shape_and_type(codec):
    code = codec.encode("il gatto mangia il topo")
    assert code.shape == codec.shape == (6, 1024)
    assert code.dtype == np.int64
    assert (code >= 0).all()


def test_codes_are_deterministic_and_stable_when_words_are_appended(corpus):
    a = Codec(["gatto", "topo", "il", "mangia"])
    b = Codec(a.words + sorted({w for s in corpus for w in s.split()} - set(a.words)))
    assert np.array_equal(a.encode("il gatto mangia il topo"), b.encode("il gatto mangia il topo"))
    assert not np.array_equal(a.encode("il gatto"), Codec(a.words, seed=1).encode("il gatto"))


def test_every_corpus_sentence_decodes_in_order(codec, corpus):
    for s in corpus:
        result = codec.decode(codec.encode(s))
        assert result.status == "exact"
        assert result.text == s


def test_random_sentences_decode_in_order(codec):
    rng = random.Random(0)
    for _ in range(50):
        words = rng.choices(codec.words, k=rng.randint(1, 30))
        result = codec.decode(codec.encode(words))
        assert result.status in ("exact", "ambiguous")
        assert codec.verify(result.words, codec.encode(words))
        if result.status == "exact":
            assert result.words == words


def test_empty_sentence(codec):
    result = codec.decode(codec.encode(""))
    assert result.status == "exact" and result.words == []


def test_repeats_make_bigrams_ambiguous_but_higher_orders_resolve_them(lp):
    s = "il gatto il cane il topo"
    two = lp.codec(orders=(2,)).decode(lp.codec(orders=(2,)).encode(s))
    assert two.status == "ambiguous"
    assert {two.text, " ".join(two.alternatives[0])} == {s, "il cane il gatto il topo"}
    assert lp.codec(orders=(2, 3)).decode(lp.codec(orders=(2, 3)).encode(s)).text == s


def test_position_channel_resolves_repeats(lp):
    s = "il gatto il cane il topo"
    codec = lp.codec(orders=(2,), positions=1)
    assert codec.decode(codec.encode(s)).status == "exact"
    assert codec.decode(codec.encode(s)).text == s
    assert lp.codec(orders=(2,), positions=0).shape == (2, 1024)


def test_words_only_codec_gives_the_bag(lp):
    codec = lp.codec(orders=())
    result = codec.decode(codec.encode("il gatto mangia il topo"))
    assert result.status == "bag"
    assert sorted(result.words) == sorted("il gatto mangia il topo".split())


def test_sum_of_two_codes_decodes_to_the_bag(codec):
    a, b = "il gatto mangia il topo", "la nonna prepara la torta di mele"
    result = codec.decode(codec.encode(a) + codec.encode(b))
    assert result.status == "bag"
    assert sorted(result.words) == sorted((a + " " + b).split())


def test_tampered_code_fails(codec):
    code = codec.encode("il gatto mangia il topo")
    code[0, np.flatnonzero(code[0])[0]] += 1
    assert codec.decode(code).status == "fail"
    assert not codec.verify("il gatto mangia il topo", code)


def test_unknown_words_are_rejected(codec):
    with pytest.raises(ValueError, match="giraffa"):
        codec.encode("il gatto e la giraffa")


def test_bad_parameters():
    with pytest.raises(ValueError):
        Codec(["a"], d=100, k=8)
    with pytest.raises(ValueError):
        Codec(["a"], orders=(2, 4))


def test_similarity_counts_shared_cells(codec):
    assert codec.similarity("gatto", "gatto") == 1.0
    assert codec.similarity("il gatto", "il gatto mangia") > codec.similarity("il gatto", "la nonna")


def test_save_and_load(tmp_path, codec, lp):
    codec.save(tmp_path)
    again = Codec.load(tmp_path)
    assert np.array_equal(again.encode("il cane dorme"), codec.encode("il cane dorme"))

    sem = lp.semantic(dim=4)
    s_codec = lp.codec(semantic=sem, k=4, d=64)
    s_codec.save(tmp_path / "s")
    assert np.array_equal(Codec.load(tmp_path / "s").encode("il cane"), s_codec.encode("il cane"))


def test_semantic_codec_still_decodes(lp, corpus):
    codec = lp.codec(semantic=lp.semantic(dim=4), k=8)
    for s in corpus:
        assert codec.decode(codec.encode(s)).text == s


def test_meaning_channel_similarity_matches_semantic(lp, corpus):
    sem = lp.semantic(dim=4)
    codec = lp.codec(semantic=sem)
    assert codec.shape == (7, 1024)
    for a in corpus[:5]:
        for b in corpus[5:10]:
            assert abs(codec.similarity(a, b) - sem.similarity(a, b)) < 1e-2


def test_meaning_channel_is_additive_and_verified(lp, corpus):
    codec = lp.codec(semantic=lp.semantic(dim=4))
    a, b = corpus[0], corpus[1]
    total = codec.encode(a) + codec.encode(b)
    assert np.array_equal(total[-1], codec.encode(a + " " + b)[-1])
    code = codec.encode(a)
    assert codec.verify(a, code)
    code[-1, 0] += 1  # a meaning that does not match the words
    assert not codec.verify(a, code)
    assert codec.decode(code).status == "bag"


def test_meaning_channel_needs_room(lp):
    with pytest.raises(ValueError, match="more than d"):
        lp.codec(semantic=lp.semantic(dim=5), d=4, k=2)


def test_semantic_codec_needs_every_word(lp):
    sem = lp.semantic(dim=4)
    with pytest.raises(ValueError, match="no semantic vector"):
        Codec(lp.words + ["giraffa"], semantic=sem)


def test_word_values_are_the_map_primes():
    from logprime import LogPrime
    lp = LogPrime()
    lp.add(["il gatto mangia il topo", "il cane dorme"])
    c = lp.codec()
    assert [int(c._val[c._index[w], 0]) for w in lp.words] == [lp.vocabulary[w] for w in lp.words]


def test_peeling_reads_words_off_the_cells(lp, corpus):
    c = lp.codec(orders=())
    for s in corpus:
        counts, rest = c._peel(c.encode(s)[0])
        assert not rest.any()
        assert {c.words[i]: int(n) for i, n in enumerate(counts) if n} == dict(Counter(s.split()))


def test_long_bag_decodes_beyond_least_squares():
    rng = np.random.default_rng(0)
    vocab = [f"w{i}" for i in range(50000)]
    c = Codec(vocab, d=1024, k=8, orders=())
    bag = sorted(rng.choice(vocab, 140, replace=False))
    v = c.encode(bag)[0]
    assert c._solve(c._pos, c._val, v) is None  # too many candidates for least squares alone
    assert sorted(c.decode(c.encode(bag)).words) == bag
