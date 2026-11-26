from logprime.parser import parse_text, STOPWORDS_IT


def test_apostrophe_keeps_right_side():
    sentences = parse_text("l'oscurità era totale.", min_words=2, stopwords=None)
    assert sentences == ["oscurità era totale"]


def test_apostrophe_variants():
    s = parse_text("dell'uomo all'alba nell'aria un'idea.", stopwords=None)[0]
    tokens = s.split()
    assert "uomo" in tokens
    assert "alba" in tokens
    assert "aria" in tokens
    assert "idea" in tokens
    # truncated parts must be gone
    assert "dell" not in tokens
    assert "all" not in tokens
    assert "nell" not in tokens
    assert "un" not in tokens


def test_apostrophe_c_e():
    # c'è → è (which may then be removed by stopwords)
    s = parse_text("c'è qualcosa.", stopwords=None)[0]
    assert "è" in s.split()
    assert "c" not in s.split()


def test_stopwords_remove_common_words():
    sentences = parse_text("io mangio la mela.", stopwords=STOPWORDS_IT)
    tokens = sentences[0].split()
    assert "la" not in tokens
    assert "mangio" in tokens
    assert "mela" in tokens


def test_min_words_filter():
    sentences = parse_text("ciao. io mangio la mela.", min_words=3, stopwords=STOPWORDS_IT)
    # "ciao" alone after stopwords → dropped; "mangio mela" = 2 tokens → dropped too
    assert all(len(s.split()) >= 3 for s in sentences)


def test_digits_removed_by_default():
    sentences = parse_text("nel 1848 tutto cambiò.", stopwords=None)
    assert "1848" not in sentences[0].split()


def test_sentence_split():
    sentences = parse_text("Prima frase. Seconda frase! Terza frase?", stopwords=None)
    assert len(sentences) == 3


def test_hard_wrapped_lines_are_joined():
    text = "la guerra\ncontinua senza fine.\n\nnuovo paragrafo qui"
    assert parse_text(text) == ["la guerra continua senza fine", "nuovo paragrafo qui"]
