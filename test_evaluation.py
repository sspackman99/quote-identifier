# test_evaluation.py
"""Tests for the labelled evaluation set and the ranking used to score it."""

import pytest

from evaluation_pairs import CONTROLS, PARAPHRASE_PAIRS, QUOTATION_PAIRS, all_pairs
from evaluate_retrieval import parse_ref, rank_chapters


def test_pairs_are_well_formed_and_unique():
    pairs = all_pairs()

    assert len(pairs) == len(QUOTATION_PAIRS) + len(PARAPHRASE_PAIRS) + len(CONTROLS)
    queries = [pair["query"] for pair in pairs]
    assert len(queries) == len(set(queries)), "duplicate query in the labelled set"

    for pair in pairs:
        book, chapter = parse_ref(pair["query"])
        assert book and isinstance(chapter, int) and chapter > 0
        if pair["kind"] == "control":
            assert pair["expected"] is None
        else:
            expected_book, expected_chapter = parse_ref(pair["expected"])
            assert expected_book and expected_chapter > 0


def test_quotation_pairs_cover_the_main_isaiah_blocks():
    pairs = {(pair["query"], pair["expected"]) for pair in QUOTATION_PAIRS}

    # spot-check the chapter-for-chapter blocks rather than trusting the file
    assert ("1 Nephi 20", "Isaiah 48") in pairs
    assert ("1 Nephi 21", "Isaiah 49") in pairs
    assert ("2 Nephi 12", "Isaiah 2") in pairs
    assert ("2 Nephi 24", "Isaiah 14") in pairs
    assert ("Mosiah 14", "Isaiah 53") in pairs
    assert ("3 Nephi 25", "Malachi 4") in pairs


def test_parse_ref_splits_on_the_last_space():
    assert parse_ref("1 Nephi 20") == ("1 Nephi", 20)
    assert parse_ref("3 Nephi 12") == ("3 Nephi", 12)
    assert parse_ref("Song of Solomon 2") == ("Song of Solomon", 2)

    with pytest.raises(ValueError):
        parse_ref("Isaiah")


def test_rank_chapters_prefers_hit_count_then_best_score():
    candidates = {
        ("Isaiah", 53): [0.91, 0.88, 0.85],
        ("Matthew", 5): [0.99],
        ("Psalms", 22): [0.90, 0.80],
    }

    ranked = rank_chapters(candidates)

    assert [(row["book"], row["chapter"]) for row in ranked] == [
        ("Isaiah", 53), ("Psalms", 22), ("Matthew", 5),
    ]
    assert ranked[0]["hits"] == 3
    assert ranked[2]["hits"] == 1


def test_rank_chapters_breaks_ties_deterministically():
    candidates = {
        ("Isaiah", 53): [0.80],
        ("Malachi", 3): [0.80],
    }

    first = rank_chapters(candidates)
    second = rank_chapters(dict(candidates))

    assert first == second
    assert [(row["book"], row["chapter"]) for row in first] == [
        ("Isaiah", 53), ("Malachi", 3),
    ]


def test_rank_chapters_handles_empty_candidates():
    assert rank_chapters({}) == []
