# test_parse_scripture.py
"""Tests for the structure-aware scripture parser in preprocess.py.

The narrow split_book_chapters/split_verses helpers only understand numbered
books with an explicit "N Book Chapter N" heading. A real corpus also contains
unnumbered books, books with no chapter heading, and inline verse markers; those
used to be swallowed into the previous chapter instead of raising, which
mislabelled most verses.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from preprocess import CORPUS_ENV_VAR, parse_scripture_text, preprocess_bom, resolve_corpus_path


def test_unnumbered_book_starts_its_own_section():
    text = """1 Nephi Chapter 1
1:1 first book verse
Mosiah Chapter 1
1:1 mosiah opening verse
"""
    verses = parse_scripture_text(text)

    assert [(v["book"], v["chapter"], v["verse"]) for v in verses] == [
        ("1 Nephi", 1, "1:1"),
        ("Mosiah", 1, "1:1"),
    ]
    assert verses[1]["text"] == "mosiah opening verse"


def test_previous_chapter_does_not_swallow_following_book():
    """The old splitter folded whole books into the last numbered chapter."""
    text = """2 Nephi Chapter 33
33:1 second nephi verse
 THE BOOK OF JACOB
Jacob Chapter 1
1:1 jacob opening verse
1:2 jacob second verse
"""
    verses = parse_scripture_text(text)

    assert [v["book"] for v in verses] == ["2 Nephi", "Jacob", "Jacob"]
    # the runaway bucket that used to collect thousands of verses
    assert len([v for v in verses if v["book"] == "2 Nephi"]) == 1


def test_single_chapter_book_without_chapter_heading():
    text = """ THE BOOK OF ENOS
1:1 behold it came to pass that i enos
1:2 and then a second verse
"""
    verses = parse_scripture_text(text)

    assert all(v["book"] == "Enos" for v in verses)
    assert all(v["chapter"] == 1 for v in verses)
    assert [v["verse"] for v in verses] == ["1:1", "1:2"]


def test_ordinal_title_with_inline_verse_markers():
    text = """3 Nephi Chapter 30
30:1 hearken o ye gentiles
 FOURTH NEPHI
4 Nephi 1:1 and it came to pass that the thirty and fourth year
4 Nephi 1:2 and it came to pass in the thirty and sixth year
 THE BOOK OF MORMON
Mormon Chapter 1
1:1 and now i mormon make a record
"""
    verses = parse_scripture_text(text)

    assert [(v["book"], v["chapter"], v["verse"]) for v in verses] == [
        ("3 Nephi", 30, "30:1"),
        ("4 Nephi", 1, "1:1"),
        ("4 Nephi", 1, "1:2"),
        ("Mormon", 1, "1:1"),
    ]


def test_front_matter_before_first_chapter_is_skipped():
    text = """The Project Gutenberg eBook of The Book of Mormon
This ebook is for the use of anyone anywhere
1 Nephi Chapter 1
1:1 real first verse
"""
    verses = parse_scripture_text(text)

    assert len(verses) == 1
    assert verses[0]["text"] == "real first verse"


def test_verse_text_continues_across_lines():
    text = """1 Nephi Chapter 1
1:1 I, Nephi, having been born of goodly parents,
therefore I was taught somewhat in all the learning
of my father.
1:2 And the first year of the reign
"""
    verses = parse_scripture_text(text)

    assert verses[0]["text"] == (
        "I, Nephi, having been born of goodly parents, "
        "therefore I was taught somewhat in all the learning of my father."
    )
    assert verses[1]["text"] == "And the first year of the reign"


def test_empty_or_headerless_text_returns_nothing():
    assert parse_scripture_text("") == []
    assert parse_scripture_text("just prose with no chapter heading") == []


# -------------------------------------------------
# Guards against the real corpus, when it is present.
# -------------------------------------------------

BOM_CANDIDATES = [
    Path(__file__).parent / "bom.txt",
    Path(__file__).parent / "data" / "bom.txt",
]


def _real_bom() -> Path:
    for candidate in BOM_CANDIDATES:
        if candidate.exists():
            return candidate
    pytest.skip("no local bom.txt to check against")


def test_real_corpus_every_verse_marker_is_parsed():
    """Verse count must equal the number of verse markers in the file."""
    path = _real_bom()
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    plain_markers = sum(
        1 for line in raw.split("\n") if line[:1].isdigit() and ":" in line.split(" ")[0]
    )
    inline_4nephi = sum(1 for line in raw.split("\n") if line.startswith("4 Nephi 1:"))

    verses = preprocess_bom(str(path))
    assert len(verses) == plain_markers + inline_4nephi


def test_real_corpus_no_runaway_chapter_bucket():
    """No (book, chapter) may hold more verses than the longest real chapter.

    The swallowing bug produced buckets of thousands of verses (e.g. 3565 under
    '2 Nephi 33'). The longest genuine Book of Mormon chapter is under 100.
    """
    path = _real_bom()
    verses = preprocess_bom(str(path))

    counts = {}
    for verse in verses:
        key = (verse["book"], verse["chapter"])
        counts[key] = counts.get(key, 0) + 1

    worst = max(counts.items(), key=lambda item: item[1])
    assert worst[1] < 100, f"runaway bucket {worst[0]} holds {worst[1]} verses"


def test_real_corpus_all_books_present():
    path = _real_bom()
    books = {verse["book"] for verse in preprocess_bom(str(path))}

    expected = {
        "1 Nephi", "2 Nephi", "Jacob", "Enos", "Jarom", "Omni", "Mosiah",
        "Alma", "Helaman", "3 Nephi", "4 Nephi", "Mormon", "Ether", "Moroni",
    }
    assert expected <= books, f"missing books: {sorted(expected - books)}"


def test_real_corpus_verse_references_are_plausible():
    """Spot-check verses whose text is well known, so mislabelling cannot pass."""
    path = _real_bom()
    verses = {(v["book"], v["chapter"], v["verse"]): v for v in preprocess_bom(str(path))}

    mosiah_1_1 = verses.get(("Mosiah", 1, "1:1"))
    assert mosiah_1_1 is not None, "Mosiah 1:1 missing"
    assert "no more contention in all the land of zarahemla" in mosiah_1_1["clean_text"]

    alma_1_1 = verses.get(("Alma", 1, "1:1"))
    assert alma_1_1 is not None, "Alma 1:1 missing"
    assert "reign of the judges" in alma_1_1["clean_text"]

    moroni_10_32 = verses.get(("Moroni", 10, "10:32"))
    assert moroni_10_32 is not None, "Moroni 10:32 missing"
    assert "deny yourselves of all ungodliness" in moroni_10_32["clean_text"]


# -------------------------------------------------
# Corpus path resolution and the CLI's failure mode.
# -------------------------------------------------


def test_resolve_uses_explicit_path(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1 Nephi Chapter 1\n1:1 text\n")

    assert resolve_corpus_path(str(corpus)) == corpus


def test_resolve_rejects_missing_explicit_path(tmp_path):
    missing = tmp_path / "nope.txt"

    with pytest.raises(FileNotFoundError, match="nope.txt"):
        resolve_corpus_path(str(missing))


def test_resolve_uses_env_var(tmp_path, monkeypatch):
    corpus = tmp_path / "from_env.txt"
    corpus.write_text("1 Nephi Chapter 1\n1:1 text\n")
    monkeypatch.setenv(CORPUS_ENV_VAR, str(corpus))

    assert resolve_corpus_path() == corpus


def test_resolve_prefers_data_dir_then_root(tmp_path, monkeypatch):
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "bom.txt").write_text("1 Nephi Chapter 1\n1:1 text\n")
    assert resolve_corpus_path() == Path("bom.txt")

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "bom.txt").write_text("1 Nephi Chapter 1\n1:1 text\n")
    assert resolve_corpus_path() == Path("data/bom.txt")


def test_resolve_error_names_where_to_put_the_file(tmp_path, monkeypatch):
    monkeypatch.delenv(CORPUS_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_corpus_path()

    message = str(excinfo.value)
    assert "data/bom.txt" in message
    assert CORPUS_ENV_VAR in message
    assert "data/README.md" in message


def test_cli_exits_2_with_clear_message_when_corpus_is_missing(tmp_path):
    result = subprocess.run(
        [sys.executable, "main.py", "--corpus", str(tmp_path / "absent.txt")],
        capture_output=True, text=True, cwd=Path(__file__).parent,
    )

    assert result.returncode == 2
    assert "absent.txt" in result.stderr
    assert "Traceback" not in result.stderr
