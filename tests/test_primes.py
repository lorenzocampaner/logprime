from itertools import islice

from logprime.primes import primes_after


def test_first_primes():
    assert list(islice(primes_after(1), 10)) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def test_resume_after_prime():
    assert list(islice(primes_after(7), 3)) == [11, 13, 17]


def test_resume_after_composite_and_large():
    assert next(primes_after(100)) == 101
    assert list(islice(primes_after(7_368_787), 2)) == [7_368_791, 7_368_811]  # 7_368_793 = 31·467·509


def test_many_primes_are_prime_and_increasing():
    ps = list(islice(primes_after(1), 5000))
    assert ps == sorted(set(ps))
    assert all(all(p % d for d in range(2, int(p**0.5) + 1)) for p in ps)
