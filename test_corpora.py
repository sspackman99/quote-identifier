# test_corpora.py
"""Tests for the canonical corpus model and the per-source parsers.

Every parser must emit the same record shape; the source layouts differ wildly
(a Gutenberg book with title lines, a tab-separated verse-per-line Bible), which
is exactly why the canonical layer exists.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from corpora import (
    CANONICAL_FIELDS,
    CORPUS_PARSERS,
    CorpusFormatError,
    available_corpora,
    canonical_book,
    format_reference,
    load_corpus_text,
    parse_bom_text,
    parse_kjv_text,
    slugify,
)
from preprocess import CORPUS_ENV_VAR, preprocess_corpus, resolve_corpus_path


# -------------------------------------------------
# Canonical model
# -------------------------------------------------

def test_slugify_builds_stable_ids():
    assert slugify("1 Nephi") == "1-nephi"
    assert slugify("Song of Solomon") == "song-of-solomon"
    assert slugify("Isaiah") == "isaiah"


def test_book_aliases_map_to_the_same_identifier():
    assert canonical_book("kjv", "Psalm") == ("psalms", "Psalms")
    assert canonical_book("kjv", "Song of Songs") == ("song-of-solomon", "Song of Solomon")
    assert canonical_book("bom", "1 Nephi his reign and ministry") == ("1-nephi", "1 Nephi")


def test_unknown_book_is_an_error_not_a_silent_record():
    with pytest.raises(CorpusFormatError):
        canonical_book("kjv", "Book of Mormon")
    with pytest.raises(CorpusFormatError):
        canonical_book("bom", "Isaiah")


def test_corpora_are_registered_with_parsers():
    assert set(available_corpora()) == {"bom", "kjv", "dc", "pgp"}
    assert set(CORPUS_PARSERS) == {"bom", "kjv", "dc", "pgp"}


def test_dc_and_pgp_books_are_registered_with_readable_ids():
    assert canonical_book("dc", "Doctrine and Covenants") == ("dc", "D&C")
    assert canonical_book("pgp", "Joseph Smith - History") == (
        "joseph-smith-history", "Joseph Smith-History",
    )


def test_unknown_corpus_is_rejected():
    with pytest.raises(ValueError, match="unknown corpus"):
        load_corpus_text("nope", "whatever")


def test_format_reference_collapses_ranges():
    assert format_reference("Isaiah", 53, 1) == "Isaiah 53:1"
    assert format_reference("Isaiah", 53, 1, 12) == "Isaiah 53:1-12"
    assert format_reference("Isaiah", 53, 1, 1) == "Isaiah 53:1"


# -------------------------------------------------
# Book of Mormon parser
# -------------------------------------------------

def test_bom_unnumbered_book_starts_its_own_section():
    text = """1 Nephi Chapter 1
1:1 first book verse
Mosiah Chapter 1
1:1 mosiah opening verse
"""
    verses = parse_bom_text(text)

    assert [(v["book_id"], v["chapter"], v["verse"]) for v in verses] == [
        ("1-nephi", 1, 1),
        ("mosiah", 1, 1),
    ]
    assert verses[1]["text"] == "mosiah opening verse"


def test_bom_previous_chapter_does_not_swallow_following_book():
    text = """2 Nephi Chapter 33
33:1 second nephi verse
 THE BOOK OF JACOB
Jacob Chapter 1
1:1 jacob opening verse
1:2 jacob second verse
"""
    verses = parse_bom_text(text)

    assert [v["book_id"] for v in verses] == ["2-nephi", "jacob", "jacob"]
    assert len([v for v in verses if v["book_id"] == "2-nephi"]) == 1


def test_bom_single_chapter_book_without_chapter_heading():
    text = """ THE BOOK OF ENOS
1:1 behold it came to pass that i enos
1:2 and then a second verse
"""
    verses = parse_bom_text(text)

    assert all(v["book_id"] == "enos" for v in verses)
    assert all(v["chapter"] == 1 for v in verses)
    assert [v["verse"] for v in verses] == [1, 2]


def test_bom_ordinal_title_with_inline_verse_markers():
    text = """3 Nephi Chapter 30
30:1 hearken o ye gentiles
 FOURTH NEPHI
4 Nephi 1:1 and it came to pass that the thirty and fourth year
4 Nephi 1:2 and it came to pass in the thirty and sixth year
 THE BOOK OF MORMON
Mormon Chapter 1
1:1 and now i mormon make a record
"""
    verses = parse_bom_text(text)

    assert [(v["book_id"], v["chapter"], v["verse"]) for v in verses] == [
        ("3-nephi", 30, 1),
        ("4-nephi", 1, 1),
        ("4-nephi", 1, 2),
        ("mormon", 1, 1),
    ]


def test_bom_words_of_mormon_is_its_own_single_chapter_book():
    text = """Omni Chapter 1
1:30 last omni verse
 THE WORDS OF MORMON
1:1 and now i mormon being about to deliver up the record
1:18 and it came to pass that after king benjamin had made an end
 THE BOOK OF MOSIAH
Mosiah Chapter 1
1:1 and now there was no more contention
"""
    verses = parse_bom_text(text)

    assert [(v["book_id"], v["verse"]) for v in verses] == [
        ("omni", 30), ("words-of-mormon", 1), ("words-of-mormon", 18), ("mosiah", 1),
    ]


def test_bom_front_matter_before_first_chapter_is_skipped():
    text = """The Project Gutenberg eBook of The Book of Mormon
This ebook is for the use of anyone anywhere
1 Nephi Chapter 1
1:1 real first verse
"""
    verses = parse_bom_text(text)

    assert len(verses) == 1
    assert verses[0]["text"] == "real first verse"


def test_bom_verse_text_continues_across_lines():
    text = """1 Nephi Chapter 1
1:1 I, Nephi, having been born of goodly parents,
therefore I was taught somewhat in all the learning
of my father.
1:2 And the first year of the reign
"""
    verses = parse_bom_text(text)

    assert verses[0]["text"] == (
        "I, Nephi, having been born of goodly parents, "
        "therefore I was taught somewhat in all the learning of my father."
    )
    assert verses[1]["text"] == "And the first year of the reign"


def test_bom_empty_or_headerless_text_returns_nothing():
    assert parse_bom_text("") == []
    assert parse_bom_text("just prose with no chapter heading") == []


# -------------------------------------------------
# King James parser
# -------------------------------------------------

KJV_FIXTURE = (
    "KJV\n"
    "King James Bible: Pure Cambridge Edition - Text courtesy of example\n"
    "Genesis 1:1\tIn the beginning God created the heaven and the earth.\n"
    "Genesis 1:2\tAnd the earth was without form, and void.\n"
    "Genesis 50:26\tSo Joseph died, being an hundred and ten years old.\n"
    "Exodus 1:1\tNow these are the names of the children of Israel.\n"
    "John 3:16\tFor God so loved the world, that he gave his only begotten Son.\n"
)


def test_kjv_parses_known_references():
    verses = {(v["book_id"], v["chapter"], v["verse"]): v for v in parse_kjv_text(KJV_FIXTURE)}

    genesis = verses[("genesis", 1, 1)]
    assert genesis["text"].startswith("In the beginning God created")
    assert genesis["source_ref"] == "Genesis 1:1"
    assert genesis["source_line"] == 3

    john = verses[("john", 3, 16)]
    assert "only begotten Son" in john["text"]


def test_kjv_handles_book_transitions():
    verses = parse_kjv_text(KJV_FIXTURE)
    order = [(v["book_id"], v["chapter"], v["verse"]) for v in verses]

    assert order == [
        ("genesis", 1, 1),
        ("genesis", 1, 2),
        ("genesis", 50, 26),
        ("exodus", 1, 1),
        ("john", 3, 16),
    ]


def test_kjv_tolerates_missing_verse_numbers():
    text = "Genesis 1:1\tfirst\nGenesis 1:3\tthird\n"
    verses = parse_kjv_text(text)

    assert [v["verse"] for v in verses] == [1, 3]


def test_kjv_rejects_a_line_that_is_neither_header_nor_verse():
    text = "KJV\nKing James Bible\nGenesis 1:1\tfirst\nthis line is not a verse at all\n"

    with pytest.raises(CorpusFormatError, match="line 4"):
        parse_kjv_text(text)


def test_kjv_rejects_a_missing_tab_separator():
    with pytest.raises(CorpusFormatError):
        parse_kjv_text("KJV\nKing James Bible\nGenesis 1:1 In the beginning\n")


def test_kjv_rejects_an_unknown_book_name():
    with pytest.raises(CorpusFormatError, match="not a known book"):
        parse_kjv_text("KJV\nKing James Bible\nHezekiah 1:1\tnot a real book\n")


def test_kjv_empty_source_is_an_error():
    with pytest.raises(CorpusFormatError):
        parse_kjv_text("KJV\nKing James Bible: nothing here\n")


# -------------------------------------------------
# Every parser emits the canonical shape
# -------------------------------------------------

@pytest.mark.parametrize("corpus_id,text", [
    ("bom", "1 Nephi Chapter 1\n1:1 a verse\n"),
    ("kjv", "KJV\nKing James Bible\nGenesis 1:1\ta verse\n"),
])
def test_records_have_exactly_the_canonical_fields(corpus_id, text):
    for record in load_corpus_text(corpus_id, text):
        assert tuple(sorted(record)) == tuple(sorted(CANONICAL_FIELDS))
        assert isinstance(record["chapter"], int)
        assert isinstance(record["verse"], int)
        assert record["book_id"]
        assert record["source_line"] >= 1


# -------------------------------------------------
# Corpus path resolution and the CLI's failure mode
# -------------------------------------------------

def test_resolve_uses_explicit_path(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("1 Nephi Chapter 1\n1:1 text\n")

    assert resolve_corpus_path(str(corpus)) == corpus


def test_resolve_rejects_missing_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        resolve_corpus_path(str(tmp_path / "nope.txt"))


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


# -------------------------------------------------
# Guards against the real corpora, when present
# -------------------------------------------------

BOM_CANDIDATES = [
    Path(__file__).parent / "data" / "bom.txt",
    Path(__file__).parent / "bom.txt",
]
KJV_CANDIDATES = [
    Path(__file__).parent / "data" / "kjv.txt",
    Path(__file__).parent / "kjv.txt",
    Path("/home/momo/work/quote-identifier/kjv.txt"),
]


def _first_existing(candidates, name):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip(f"no local {name} to check against")


def test_real_bom_every_verse_marker_is_parsed():
    path = _first_existing(BOM_CANDIDATES, "bom.txt")
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")

    plain_markers = sum(
        1 for line in raw.split("\n") if line[:1].isdigit() and ":" in line.split(" ")[0]
    )
    inline_4nephi = sum(1 for line in raw.split("\n") if line.startswith("4 Nephi 1:"))

    verses = preprocess_corpus("bom", str(path))
    assert len(verses) == plain_markers + inline_4nephi


def test_real_bom_no_runaway_chapter_bucket():
    path = _first_existing(BOM_CANDIDATES, "bom.txt")
    verses = preprocess_corpus("bom", str(path))

    counts = {}
    for verse in verses:
        key = (verse["book"], verse["chapter"])
        counts[key] = counts.get(key, 0) + 1

    worst = max(counts.items(), key=lambda item: item[1])
    assert worst[1] < 100, f"runaway bucket {worst[0]} holds {worst[1]} verses"


def test_real_bom_per_book_counts_match_the_source():
    path = _first_existing(BOM_CANDIDATES, "bom.txt")
    verses = preprocess_corpus("bom", str(path))

    expected = {
        "1 Nephi": 617, "2 Nephi": 779, "Jacob": 203, "Enos": 27, "Jarom": 15,
        "Omni": 30, "Words of Mormon": 18, "Mosiah": 785, "Alma": 1973,
        "Helaman": 496, "3 Nephi": 783, "4 Nephi": 49, "Mormon": 227,
        "Ether": 433, "Moroni": 163,
    }
    counts = {}
    for verse in verses:
        counts[verse["book"]] = counts.get(verse["book"], 0) + 1

    assert counts == expected


def test_real_bom_verse_references_are_plausible():
    path = _first_existing(BOM_CANDIDATES, "bom.txt")
    verses = {(v["book"], v["chapter"], v["verse"]): v
              for v in preprocess_corpus("bom", str(path))}

    assert "no more contention in all the land of zarahemla" in \
        verses[("Mosiah", 1, 1)]["clean_text"]
    assert "reign of the judges" in verses[("Alma", 1, 1)]["clean_text"]
    assert "deny yourselves of all ungodliness" in \
        verses[("Moroni", 10, 32)]["clean_text"]


def test_real_kjv_book_and_verse_counts():
    path = _first_existing(KJV_CANDIDATES, "kjv.txt")
    verses = preprocess_corpus("kjv", str(path))

    books = {verse["book_id"] for verse in verses}
    assert len(books) == 66
    assert len(verses) == 31102

    indexed = {(v["book_id"], v["chapter"], v["verse"]): v for v in verses}
    assert indexed[("genesis", 1, 1)]["text"].startswith("In the beginning God created")
    assert "only begotten Son" in indexed[("john", 3, 16)]["text"]
    assert indexed[("isaiah", 53, 1)]["text"].startswith("Who hath believed")
    assert indexed[("malachi", 4, 6)]["book"] == "Malachi"
    assert indexed[("matthew", 1, 1)]["book"] == "Matthew"


def test_gutenberg_licence_is_not_parsed_as_scripture():
    """The footer block must not be appended to Moroni 10:34.

    Left in, the licence turns the final verse into hundreds of lines and feeds
    "project gutenberg" windows into retrieval and into the classifier.
    """
    text = (
        "THE FOURTH BOOK OF NEPHI (4 Nephi)\n\n"
        "4 Nephi Chapter 1\n\n"
        "1:1 And it came to pass that the disciples went forth.\n"
        "1:2 And they did all things common among them.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK THE BOOK OF MORMON ***\n\n"
        "Section 1. General Terms of Use\n"
        "Project Gutenberg is a registered trademark.\n"
    )

    verses = CORPUS_PARSERS["bom"](text)

    last = verses[-1]
    assert last["verse"] == 2
    assert "common among them" in last["text"]
    assert "gutenberg" not in last["text"].lower()
    assert "trademark" not in last["text"].lower()
    assert len(verses) == 2


def test_parse_is_unchanged_when_there_is_no_licence_marker():
    text = (
        "THE FOURTH BOOK OF NEPHI (4 Nephi)\n\n"
        "4 Nephi Chapter 1\n\n"
        "1:1 And it came to pass that the disciples went forth.\n"
    )

    verses = CORPUS_PARSERS["bom"](text)

    assert len(verses) == 1
    assert "disciples went forth" in verses[0]["text"]


DC_SAMPLE = """\
Doctrine and Covenants of the Church of Jesus Christ of the Latter-Day Saints, at sacred-texts.com

THE
 DOCTRINE AND COVENANTS

1. Revelation through Joseph Smith, Hiram, Ohio, November 1, 1831.

[1:1] Hearken, O ye people of my church, saith the voice of him who dwells on high.
[1:2] For verily the voice of the Lord is unto all men.

Doctrine and Covenants of the Church of Jesus Christ of the Latter-Day Saints, at sacred-texts.com

Revelation through Joseph Smith, Hiram, Ohio, March 1832. See
History of the Church 1:257.

[2:1] Hearken, O ye elders of my church, and give ear to my word.
"""

PGP_SAMPLE = """\
Pearl of Great Price: Moses: Chapter 1
1 The words of God, which he spake unto Moses at a time when Moses was caught up.
2 And he saw God face to face, and he talked with him, and the glory of God was upon Moses;
because of the transgression of my people.

Sacred Texts
|
Mormonism
Next: Pearl of Great Price: Moses: Chapter 2 »

Pearl of Great Price: Moses: Chapter 2
3 And it came to pass that Moses looked upon Satan and said: Who art thou?
4 And now, Moses, my son, I will speak unto thee concerning this earth.

« Previous: Pearl of Great Price: Moses: Chapter 1
Index

Pearl of Great Price: Moses: Chapter 1
1 The words of God, which he spake unto Moses at a time when Moses was caught up.
2 And he saw God face to face, and he talked with him, and the glory of God was upon Moses;
"""


def test_dc_verses_are_read_from_bracket_markers():
    verses = CORPUS_PARSERS["dc"](DC_SAMPLE)

    assert [(v["book"], v["chapter"], v["verse"]) for v in verses] == [
        ("D&C", 1, 1), ("D&C", 1, 2), ("D&C", 2, 1),
    ]
    assert "Hearken, O ye people" in verses[0]["text"]
    assert verses[0]["book_id"] == "dc"


def test_dc_unnumbered_and_wrapped_headings_are_tolerated():
    """Section 78 lost its number in this edition; headings also wrap."""
    verses = CORPUS_PARSERS["dc"](DC_SAMPLE)

    assert verses[-1]["chapter"] == 2, "the unnumbered heading still yields section 2"
    assert "elders of my church" in verses[-1]["text"]


def test_dc_verse_without_its_bracket_raises():
    broken = DC_SAMPLE.replace("[1:2] For verily", "1:2 For verily")

    with pytest.raises(CorpusFormatError) as error:
        CORPUS_PARSERS["dc"](broken)

    assert "bracket" in str(error.value)


def test_pgp_headers_verses_and_wrapped_lines():
    verses = CORPUS_PARSERS["pgp"](PGP_SAMPLE)

    assert [(v["book"], v["chapter"], v["verse"]) for v in verses] == [
        ("Moses", 1, 1), ("Moses", 1, 2), ("Moses", 2, 3), ("Moses", 2, 4),
    ]
    assert "because of the transgression" in verses[1]["text"], "wrapped line joined"


def test_pgp_duplicate_chapter_block_is_skipped():
    verses = CORPUS_PARSERS["pgp"](PGP_SAMPLE)

    assert sum(1 for v in verses if v["chapter"] == 1) == 2, "the repeat adds nothing"


def test_pgp_navigation_lines_are_not_text():
    verses = CORPUS_PARSERS["pgp"](PGP_SAMPLE)

    for verse in verses:
        assert "Sacred Texts" not in verse["text"]
        assert "Mormonism" not in verse["text"]
        assert "Next:" not in verse["text"]


def test_pgp_double_encoded_characters_are_repaired():
    """The scrape double-encodes punctuation: '’' arrives as 'â€™'."""
    text = PGP_SAMPLE.replace("face to face", "face to face, Godâ€™s word")

    verses = CORPUS_PARSERS["pgp"](text)

    assert "God’s word" in verses[1]["text"]
    assert "â€" not in verses[1]["text"]


DATA = Path(__file__).parent / "data"


def test_real_doctrine_and_covenants_parses_completely():
    path = DATA / "dc.txt"
    if not path.exists():
        pytest.skip("no dc.txt")
    verses = load_corpus_text("dc", path.read_text(encoding="utf-8-sig"))

    assert len(verses) == 3654
    assert len({v["chapter"] for v in verses}) == 138
    assert verses[0]["chapter"] == 1 and verses[0]["verse"] == 1
    assert verses[-1]["chapter"] == 138


def test_real_pearl_of_great_price_parses_completely():
    path = DATA / "pgp.txt"
    if not path.exists():
        pytest.skip("no pgp.txt")
    verses = load_corpus_text("pgp", path.read_text(encoding="utf-8-sig"))

    assert len(verses) == 622
    for book in ("Moses", "Abraham", "Joseph Smith-Matthew", "Joseph Smith-History"):
        assert any(v["book"] == book for v in verses), book
    assert sum(1 for v in verses if v["book"] == "Joseph Smith-History") == 75
    assert not any("Ã" in v["text"] for v in verses)
