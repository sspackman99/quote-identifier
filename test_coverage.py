# test_coverage.py
"""Tests for the deterministic extent measurement."""

import pytest

from coverage import (
    NGRAM_N,
    REPRODUCED_VERSE_THRESHOLD,
    extent_reading,
    measure,
    ngram_set,
    tokenise,
    verse_containment,
)

PSALM = [
    (1, "O god thou hast cast us off thou hast scattered us"),
    (2, "thou hast been displeased o turn thyself to us again"),
    (3, "thou hast made the earth to tremble thou hast broken it"),
    (4, "thou hast shewed thy people hard things"),
    (5, "thou hast given us a cup of trembling to drink"),
    (6, "thou wilt turn and cause thine anger toward us to cease"),
]


def test_tokenise_lowercases_and_drops_punctuation():
    assert tokenise("O God, Thou hast: [cast] us off!") == [
        "o", "god", "thou", "hast", "cast", "us", "off",
    ]


def test_ngram_set_returns_empty_for_short_text():
    assert ngram_set(["one", "two"]) == set()


def test_verse_containment_is_one_for_an_identical_verse():
    verse = PSALM[0][1]
    assert verse_containment(verse, ngram_set(tokenise(verse))) == 1.0


def test_a_whole_chapter_reproduced_reads_as_quotation():
    query = " ".join(text for _, text in PSALM)

    measurement = measure(PSALM, query)
    reading = extent_reading(measurement)

    assert measurement["reproduced_share"] == 1.0
    assert measurement["longest_run"] == len(PSALM)
    assert reading["extent"] == "quotation"


def test_two_verses_of_six_reads_as_partial_quotation():
    query = PSALM[5][1]                     # only the last verse is reproduced

    measurement = measure(PSALM, query)
    reading = extent_reading(measurement)

    assert measurement["reproduced_count"] == 1
    assert measurement["reproduced_share"] < 0.60
    assert reading["extent"] == "partial_quotation"
    assert "excerpt" in reading["reason"]


def test_a_reworded_restatement_reads_as_no_reproduction():
    query = ("god you have rejected us and scattered us you were angry with us "
             "please restore us you shook the land and split it open")

    measurement = measure(PSALM, query)

    assert measurement["reproduced_count"] == 0
    assert extent_reading(measurement)["extent"] == "no reproduction"


def test_a_short_source_quoted_whole_is_still_a_quotation():
    """The psalm case: two verses, both quoted, is not 'partial' because it is short."""
    short_source = PSALM[0:2]
    query = " ".join(text for _, text in short_source)

    reading = extent_reading(measure(short_source, query))

    assert reading["extent"] == "quotation"


def test_query_share_reports_how_much_of_the_query_comes_from_the_source():
    query = PSALM[0][1] + " and then a long stretch of unrelated narrative follows here"

    measurement = measure(PSALM, query)

    assert 0 < measurement["query_share"] < 1


def test_measure_requires_source_verses():
    with pytest.raises(ValueError):
        measure([], "anything")


def test_thresholds_are_named_not_magic():
    assert NGRAM_N == 4
    assert REPRODUCED_VERSE_THRESHOLD == 0.50
