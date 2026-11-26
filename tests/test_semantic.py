import numpy as np

from logprime import LogPrime, Semantic
from logprime.semantic import cooccurrence


def test_cooccurrence_counts_are_symmetric_and_exact():
    lp = LogPrime().add(["a b c", "a b", "c a"])
    C = cooccurrence(lp).toarray()
    i = lp._position
    assert (C == C.T).all()
    assert C[i["a"], i["b"]] == 2
    assert C[i["a"], i["c"]] == 2
    assert C[i["b"], i["c"]] == 1
    assert (np.diag(C) == 0).all()


def test_vectors_are_unit_length(lp):
    sem = lp.semantic(dim=5)
    assert sem.vectors.shape == (len(lp), 5)
    assert np.allclose(np.linalg.norm(sem.vectors, axis=1), 1, atol=1e-5)


def test_similarity_and_neighbours(lp):
    sem = lp.semantic(dim=5)
    assert abs(sem.similarity("gatto", "gatto") - 1) < 1e-5
    assert -1 <= sem.similarity("gatto", "nonna") <= 1
    assert sem.similarity("gatto", "giraffa") == 0.0
    near = sem.neighbours("gatto", n=3, min_count=1)
    assert len(near) == 3 and "gatto" not in [w for w, _ in near]


def test_deterministic(lp):
    assert np.array_equal(lp.semantic(dim=5).vectors, lp.semantic(dim=5).vectors)


def test_save_and_load(tmp_path, lp):
    sem = lp.semantic(dim=5)
    sem.save(tmp_path / "sem.npz")
    again = Semantic.load(tmp_path / "sem.npz")
    assert again.words == sem.words
    assert np.array_equal(again.vectors, sem.vectors)
