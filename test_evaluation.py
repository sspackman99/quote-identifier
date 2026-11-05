# test_evaluation.py
"""Tests for the labelled evaluation set and the scoring used on it."""

from pathlib import Path

import pytest

from evaluation_pairs import (
    CONTEXT_KINDS,
    CONTROL_KINDS,
    FAMILY_SPLITS,
    RETRIEVAL_KINDS,
    all_pairs,
    controls,
    positives,
    retrieval_positives,
    split_for_family,
)
from evaluate_retrieval import (
    parse_ref,
    rank_chapters,
    recall_metrics,
    threshold_metrics,
)
from preprocess import preprocess_corpus


# -------------------------------------------------
# The labelled set
# -------------------------------------------------

def test_pairs_are_well_formed_and_unique():
    pairs = all_pairs()
    assert len(pairs) == len(positives()) + len(controls())

    queries = [pair["query"] for pair in pairs]
    assert len(queries) == len(set(queries)), "duplicate query in the labelled set"

    for pair in pairs:
        book, chapter = parse_ref(pair["query"])
        assert book and isinstance(chapter, int) and chapter > 0
        assert pair["kind"] in RETRIEVAL_KINDS + CONTEXT_KINDS + CONTROL_KINDS
        assert pair["split"] in {"dev", "test"}
        assert pair["query_span"] == "whole" or pair["query_span"][0].isdigit()
        assert pair["expected_span"] == "whole" or pair["expected_span"][0].isdigit()
        if pair["kind"] in CONTROL_KINDS:
            assert pair["expected"] is None
        else:
            expected_book, expected_chapter = parse_ref(pair["expected"])
            assert expected_book and expected_chapter > 0
            assert pair["confidence"] in {"high", "medium", "low"}


def test_every_family_has_a_split():
    families_in_pairs = {pair["family"] for pair in all_pairs()}

    missing = families_in_pairs - set(FAMILY_SPLITS)
    assert not missing, f"families without a split: {sorted(missing)}"

    with pytest.raises(KeyError):
        split_for_family("no-such-family")


def test_a_source_family_never_straddles_the_split():
    """Neighbouring chapters of one quotation block stay on the same side."""
    by_family = {}
    for pair in all_pairs():
        by_family.setdefault(pair["family"], set()).add(pair["split"])

    straddling = {family: splits for family, splits in by_family.items() if len(splits) > 1}
    assert not straddling, f"families split across dev/test: {straddling}"


def test_both_splits_contain_positives_and_controls():
    for split in ("dev", "test"):
        rows = [pair for pair in all_pairs() if pair["split"] == split]
        assert any(pair["kind"] in RETRIEVAL_KINDS for pair in rows)
        assert any(pair["kind"] in CONTROL_KINDS for pair in rows)


def test_interpretation_is_not_counted_as_retrieval():
    interp = [pair for pair in all_pairs() if pair["kind"] in CONTEXT_KINDS]
    assert interp, "the contextual cases should still be labelled"

    retrieval_queries = {pair["query"] for pair in retrieval_positives()}
    assert not {pair["query"] for pair in interp} & retrieval_queries

    # Mosiah 15 is Abinadi explaining the passage he just quoted
    mosiah15 = next(pair for pair in interp if pair["query"] == "Mosiah 15")
    assert "Mosiah 14" in mosiah15["note"]


def test_reviewed_labels_are_marked_and_notes_explain_judgements():
    reviewed = [pair for pair in all_pairs() if pair["reviewed"]]

    for query in ("Alma 7", "Mosiah 15", "3 Nephi 19", "Mosiah 12", "3 Nephi 20",
                  "Moroni 7", "Mosiah 4"):
        pair = next(pair for pair in all_pairs() if pair["query"] == query)
        assert pair["reviewed"], f"{query} should be marked reviewed"

    assert all(pair["note"] for pair in reviewed)


def test_3_nephi_20_records_the_corrected_span():
    pair = next(p for p in all_pairs() if p["query"] == "3 Nephi 20")

    assert pair["query_span"] == "36-40"
    assert "23-26" in pair["note"] and "NOT" in pair["note"]


def test_former_shared_phrase_controls_are_now_positives():
    """Moroni 10, 2 Nephi 2 and Alma 42 assert relationships; none is a control."""
    by_query = {pair["query"]: pair for pair in all_pairs()}

    assert by_query["Moroni 10"]["kind"] == "partial_quotation"
    assert by_query["Moroni 10"]["expected"] == "Isaiah 52"
    assert by_query["2 Nephi 2"]["kind"] == "interpretation"
    assert by_query["Alma 42"]["kind"] == "interpretation"
    assert by_query["Mosiah 4"]["kind"] == "shared_vocabulary"


# -------------------------------------------------
# Span validity against the real corpora
# -------------------------------------------------

BOM_CANDIDATES = [Path(__file__).parent / "data" / "bom.txt",
                  Path(__file__).parent / "bom.txt"]
KJV_CANDIDATES = [Path(__file__).parent / "data" / "kjv.txt",
                  Path(__file__).parent / "kjv.txt",
                  Path("/home/momo/work/quote-identifier/kjv.txt")]


def _corpus_index(candidates, corpus_id, name):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        pytest.skip(f"no local {name} to check spans against")

    index = {}
    for verse in preprocess_corpus(corpus_id, str(path)):
        index.setdefault((verse["book"], verse["chapter"]), set()).add(verse["verse"])
    return index


def _expand(span):
    """'1-2,7-10' -> {1, 2, 7, 8, 9, 10}"""
    numbers = set()
    for part in span.split(","):
        if "-" in part:
            start, end = part.split("-")
            numbers.update(range(int(start), int(end) + 1))
        else:
            numbers.add(int(part))
    return numbers


def test_query_spans_exist_in_the_book_of_mormon():
    index = _corpus_index(BOM_CANDIDATES, "bom", "bom.txt")

    for pair in all_pairs():
        if pair["query_span"] == "whole":
            continue
        book, chapter = parse_ref(pair["query"])
        missing = _expand(pair["query_span"]) - index[(book, chapter)]
        assert not missing, f"{pair['query']} has no verses {sorted(missing)}"


def test_expected_spans_exist_in_the_king_james_bible():
    index = _corpus_index(KJV_CANDIDATES, "kjv", "kjv.txt")

    for pair in all_pairs():
        if pair["expected_span"] == "whole" or not pair["expected"]:
            continue
        book, chapter = parse_ref(pair["expected"])
        missing = _expand(pair["expected_span"]) - index[(book, chapter)]
        assert not missing, f"{pair['expected']} has no verses {sorted(missing)}"


# -------------------------------------------------
# Scoring
# -------------------------------------------------

def test_parse_ref_splits_on_the_last_space():
    assert parse_ref("1 Nephi 20") == ("1 Nephi", 20)
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


def test_rank_chapters_breaks_ties_deterministically():
    candidates = {("Isaiah", 53): [0.80], ("Malachi", 3): [0.80]}

    assert rank_chapters(candidates) == rank_chapters(dict(candidates))
    assert [(row["book"], row["chapter"]) for row in rank_chapters(candidates)] == [
        ("Isaiah", 53), ("Malachi", 3),
    ]


def test_rank_chapters_handles_empty_candidates():
    assert rank_chapters({}) == []


def _row(query, expected, rank, top1, top1_best, kind="quotation", split="dev"):
    # both ranking rules are carried on a result row; these tests exercise the
    # metric arithmetic, so the two ranks are set to the same value
    return {"query": query, "expected": expected, "kind": kind, "split": split,
            "expected_rank": rank, "chapter_rank": rank, "span_rank": rank,
            "top1": top1, "top1_best": top1_best}


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


def test_recall_ignores_interpretation_rows():
    rows = [
        _row("a", "Isaiah 48", 1, "Isaiah 48", 0.99),
        _row("b", "Isaiah 53", 900, "Luke 1", 0.60, kind="interpretation"),
    ]

    metrics = recall_metrics(rows)

    assert metrics["pairs"] == 1
    assert metrics["recall_at_1"] == 1.0


def test_threshold_metrics_separates_controls_from_context():
    rows = [
        _row("a", "Isaiah 48", 1, "Isaiah 48", 0.99),                      # tp
        _row("b", "Isaiah 49", 4, "Psalms 2", 0.95),                       # wrong top hit
        _row("c", None, None, "Judges 11", 0.93, kind="unrelated"),        # control FP
        _row("d", "Genesis 3", 1, "Genesis 3", 0.97, kind="interpretation"),
        _row("e", None, None, "Luke 3", 0.60, kind="unrelated"),
    ]

    metrics = threshold_metrics(rows, 0.90)

    assert metrics["accepted"] == 4
    assert metrics["true_positives"] == 1
    assert metrics["false_positives_controls"] == 1
    assert metrics["accepted_context"] == 1
    assert metrics["false_positives_other"] == 1
    assert metrics["precision"] == pytest.approx(1 / 4)
    assert metrics["recall"] == pytest.approx(1 / 2)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 2)


def test_threshold_metrics_returns_none_without_scores():
    assert threshold_metrics([], 0.9) is None
