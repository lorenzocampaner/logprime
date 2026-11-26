from logprime import LogPrime
from logprime.cli import main


def test_build_and_add(tmp_path):
    (tmp_path / "a.txt").write_text("io mangio la mela\n\nio compro la mela\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("tu mangi la pera\n", encoding="utf-8")
    out = tmp_path / "map"

    main(["build", str(tmp_path / "a.txt"), "-o", str(out)])
    assert len(LogPrime.load(out)) == 5

    main(["build", str(tmp_path / "b.txt"), "-o", str(out), "--add"])
    lp = LogPrime.load(out)
    assert len(lp) == 8
    assert lp.vocabulary["io"] == 2


def test_build_directory_with_parser(tmp_path):
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "a.txt").write_text("L'uomo mangia.\nLa donna\nbeve il tè.", encoding="utf-8")
    main(["build", str(texts), "-o", str(tmp_path / "map"), "--parse"])
    assert set(LogPrime.load(tmp_path / "map").vocabulary) == {"uomo", "mangia", "la", "donna", "beve", "il", "tè"}


def test_info_and_decode(tmp_path, capsys, corpus):
    LogPrime().add(corpus).save(tmp_path)
    main(["info", str(tmp_path), "--word", "gatto"])
    main(["decode", str(tmp_path), "il gatto mangia il topo"])
    out = capsys.readouterr().out
    assert "prime" in out
    assert "[ok] verified" in out


def test_code_and_semantic(tmp_path, capsys, corpus):
    LogPrime().add(corpus).save(tmp_path)
    main(["code", str(tmp_path), "il gatto mangia il topo"])
    main(["code", str(tmp_path), "il gatto il cane il topo", "--max-n", "2"])
    main(["semantic", str(tmp_path), "--dim", "4", "-w", "gatto", "--min-count", "1"])
    out = capsys.readouterr().out
    assert "[ok] exact" in out
    assert "[ambiguous]" in out
    assert (tmp_path / "semantic.npz").exists()
    assert out.strip().splitlines()[-1].startswith("gatto")
