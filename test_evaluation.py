# test_evaluation.py
"""Tests for the labelled evaluation set and the scoring used on it."""

import pytest

from evaluation_pairs import (
    ALLUSION_PAIRS,
    PARAPHRASE_PAIRS,
    PARTIAL_QUOTATION_PAIRS,
    QUOTATION_PAIRS,
    SHARED_PHRASE_CONTROLS,
    UNRELATED_CONTROLS,
    all_pairs,
    controls,
    positives,
    split_for,
)
from evaluate_retrieval import (
    parse_ref,
    rank_chapters,
    recall_metrics,
    threshold_metrics,
)


# -------------------------------------------------
# The labelled set
# -------------------------------------------------

def test_pairs_are_well_formed_and_unique():
    pairs = all_pairs()

    assert len(pairs) == (
        len(QUOTATION_PAIRS) + len(PARTIAL_QUOTATION_PAIRS) + len(PARAPHRASE_PAIRS)
        + len(ALLUSION_PAIRS) + len(SHARED_PHRASE_CONTROLS) + len(UNRELATED_CONTROLS)
    )
    queries = [pair["query"] for pair in pairs]
    assert len(queries) == len(set(queries)), "duplicate query in the labelled set"

    for pair in pairs:
        book, chapter = parse_ref(pair["query"])
        assert book and isinstance(chapter, int) and chapter > 0
        assert pair["kind"] in {
            "quotation", "partial_quotation", "paraphrase", "allusion",
            "shared_phrases", "unrelated",
        }
        assert pair["split"] in {"dev", "test"}
        if pair["expected"]:
            expected_book, expected_chapter = parse_ref(pair["expected"])
            assert expected_book and expected_chapter > 0
            assert pair["confidence"] in {"high", "medium", "low"}
        else:
            assert pair["kind"] in {"shared_phrases", "unrelated"}


def test_quotation_pairs_cover_the_main_isaiah_blocks():
    pairs = {(pair["query"], pair["expected"]) for pair in QUOTATION_PAIRS}

    assert ("1 Nephi 20", "Isaiah 48") in pairs
    assert ("1 Nephi 21", "Isaiah 49") in pairs
    assert ("2 Nephi 12", "Isaiah 2") in pairs
    assert ("2 Nephi 24", "Isaiah 14") in pairs
    assert ("Mosiah 14", "Isaiah 53") in pairs
    assert ("3 Nephi 25", "Malachi 4") in pairs


def test_controls_have_no_expected_chapter():
    assert all(row["expected"] is None for row in controls())
    assert all(row["expected"] for row in positives())


def test_split_is_deterministic_and_both_sides_are_populated():
    assert split_for("Mosiah 14") == split_for("Mosiah 14")
    assert {pair["split"] for pair in all_pairs()} == {"dev", "test"}

    # both splits need positives and controls, or the held-out number says
    # nothing about either
    for split in ("dev", "test"):
        rows = [pair for pair in all_pairs() if pair["split"] == split]
        assert any(row["expected"] for row in rows)
        assert any(not row["expected"] for row in rows)


def test_low_confidence_labels_carry_a_justification():
    allusions = {row["query"]: row for row in ALLUSION_PAIRS}

    assert allusions["Alma 7"]["confidence"] == "low"
    assert allusions["Alma 7"]["note"]
    assert all(row["confidence"] == "low" for row in ALLUSION_PAIRS)


# -------------------------------------------------
# Scoring
# -------------------------------------------------

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
    candidates = {("Isaiah", 53): [0.80], ("Malachi", 3): [0.80]}

    first = rank_chapters(candidates)
    second = rank_chapters(dict(candidates))

    assert first == second
    assert [(row["book"], row["chapter"]) for row in first] == [
        ("Isaiah", 53), ("Malachi", 3),
    ]


def test_rank_chapters_handles_empty_candidates():
    assert rank_chapters({}) == []


def _row(query, expected, rank, top1, top1_best, kind="quotation", split="dev"):
    return {"query": query, "expected": expected, "kind": kind, "split": split,
            "expected_rank": rank, "top1": top1, "top1_best": top1_best}


def test_recall_counts_ranks_within_cutoffs():
    rows = [
        _row("a", "Isaiah 48", 1, "Isaiah 48", 0.99),
        _row("b", "Isaiah 49", 3, "Psalms 2", 0.92),
        _row("c", "Isaiah 50", None, "Psalms 3", 0.90),
    ]

    metrics = recall_metrics(rows)

    assert metrics["pairs"] == 3
    assert metrics["recall_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["recall_at_20"] == pytest.approx(2 / 3)
    assert metrics["missed"] == ["c"]


def test_threshold_metrics_counts_controls_as_false_positives():
    rows = [
        _row("a", "Isaiah 48", 1, "Isaiah 48", 0.99),          # true positive
        _row("b", "Isaiah 49", 4, "Psalms 2", 0.95),           # right pair, wrong top hit
        _row("c", None, None, "Judges 11", 0.93, kind="unrelated"),
        _row("d", None, None, "Luke 3", 0.60, kind="unrelated"),
    ]

    metrics = threshold_metrics(rows, 0.90)

    assert metrics["accepted"] == 3
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 2)


def test_threshold_metrics_returns_none_without_scores():
    assert threshold_metrics([], 0.9) is None
