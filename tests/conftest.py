import pytest

from logprime import LogPrime

# A small hand-written corpus, so the tests do not depend on external texts.
CORPUS = [
    "il gatto mangia il topo",
    "il cane insegue il gatto",
    "la ragazza legge un libro in giardino",
    "il ragazzo scrive una lettera alla nonna",
    "la nonna prepara la torta di mele",
    "il treno parte dalla stazione alle otto",
    "la pioggia cade sulla città grigia",
    "il sole splende sopra il mare calmo",
    "un pescatore ripara la rete sulla spiaggia",
    "il medico visita il paziente in ospedale",
    "la maestra spiega la lezione agli alunni",
    "il vento muove le foglie del bosco",
    "il cane dorme vicino al camino",
    "la ragazza beve il tè con la nonna",
    "il pescatore vende il pesce al mercato",
    "il treno arriva in città con la pioggia",
]


@pytest.fixture
def corpus():
    return list(CORPUS)


@pytest.fixture
def lp(corpus):
    return LogPrime().add(corpus)
