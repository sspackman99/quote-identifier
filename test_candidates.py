# test_candidates.py
"""Tests for span-aware candidate scoring.

These are unit tests on synthetic metadata: the point is the ranking rule and the
evidence it keeps, not the corpus.
"""

from candidates import (
    aggregate_candidates,
    format_ranges,
    merge_intervals,
    query_span_hit,
    rank_candidates,
    span_hit,
    verses_per_chapter,
)


def window(book, chapter, start_verse, end_verse, corpus="kjv"):
    return {"corpus": corpus, "book_id": book.lower(), "book": book,
            "chapter": chapter, "start_verse": start_verse, "end_verse": end_verse,
            "text": ""}


def match(sim, query_index, target_index):
    return {"sim": sim, "query_index": query_index, "target_index": target_index}


def test_merge_intervals_collapses_adjacent_and_overlapping_ranges():
    assert merge_intervals([(1, 2), (2, 3), (4, 5)]) == [(1, 3), (4, 5)]
    assert merge_intervals([(5, 5)]) == [(5, 5)]
    assert merge_intervals([]) == []


def test_format_ranges_prints_readable_spans():
    assert format_ranges([(1, 2), (7, 10)]) == "1-2,7-10"
    assert format_ranges([(5, 5)]) == "5"


def test_verses_per_chapter_counts_distinct_verses():
    meta = [window("Isaiah", 52, 1, 2), window("Isaiah", 52, 2, 3),
            window("Isaiah", 52, 3, 4)]

    assert verses_per_chapter(meta) == {("Isaiah", 52): 4}


def test_overlapping_windows_count_as_one_support():
    meta = [
        window("Mosiah", 14, 1, 2, corpus="bom"),   # query window 0
        window("Isaiah", 53, 1, 2),                 # target 1
        window("Isaiah", 53, 2, 3),                 # target 2 (overlaps 1)
        window("Isaiah", 53, 4, 5),                 # target 3 (separate)
    ]
    matches = [match(0.9, 0, 1), match(0.8, 0, 2), match(0.7, 0, 3)]

    candidates = aggregate_candidates(meta, matches)

    candidate = candidates[("Isaiah", 53)]
    assert candidate["support"] == 2          # (1-3) and (4-5)
    assert candidate["best"] == 0.9
    assert candidate["source_range"] == "1-3,4-5"
    assert candidate["matches"] == 3


def test_coverage_is_share_of_the_candidate_chapter():
    meta = [
        window("Mosiah", 14, 1, 2, corpus="bom"),
        window("Isaiah", 52, 1, 2),
        window("Isaiah", 52, 2, 3),
        window("Isaiah", 52, 3, 4),
        window("Isaiah", 52, 4, 5),
    ]
    matches = [match(0.9, 0, 1)]

    candidate = aggregate_candidates(meta, matches)[("Isaiah", 52)]

    # the chapter's windows cover verses 1-5; the matched range covers 1-2
    assert candidate["coverage"] == 0.4


def test_a_lone_high_score_does_not_beat_real_support_within_tolerance():
    """The rule: best similarity leads, but a near-tie defers to support."""
    candidates = {
        ("Psalms", 22): {"book": "Psalms", "chapter": 22, "best": 0.97, "support": 1,
                         "coverage": 0.1, "source_intervals": [(1, 2)],
                         "query_intervals": [(1, 2)], "matches": 1},
        ("Isaiah", 53): {"book": "Isaiah", "chapter": 53, "best": 0.965, "support": 4,
                         "coverage": 0.6, "source_intervals": [(1, 4)],
                         "query_intervals": [(1, 4)], "matches": 6},
    }

    ranked = rank_candidates(candidates, tie_tolerance=0.02)

    assert [(row["book"], row["chapter"]) for row in ranked] == [
        ("Isaiah", 53), ("Psalms", 22),
    ]


def test_a_clearly_higher_score_still_wins_outside_tolerance():
    candidates = {
        ("Psalms", 22): {"book": "Psalms", "chapter": 22, "best": 0.99, "support": 1,
                         "coverage": 0.1, "source_intervals": [(1, 2)],
                         "query_intervals": [(1, 2)], "matches": 1},
        ("Isaiah", 53): {"book": "Isaiah", "chapter": 53, "best": 0.90, "support": 5,
                         "coverage": 0.8, "source_intervals": [(1, 5)],
                         "query_intervals": [(1, 5)], "matches": 8},
    }

    ranked = rank_candidates(candidates, tie_tolerance=0.02)

    assert ranked[0]["book"] == "Psalms"


def test_ranking_is_deterministic_on_a_full_tie():
    candidates = {
        ("Isaiah", 53): {"book": "Isaiah", "chapter": 53, "best": 0.9, "support": 2,
                         "coverage": 0.5, "source_intervals": [(1, 3)],
                         "query_intervals": [(1, 3)], "matches": 2},
        ("Malachi", 3): {"book": "Malachi", "chapter": 3, "best": 0.9, "support": 2,
                         "coverage": 0.5, "source_intervals": [(1, 3)],
                         "query_intervals": [(1, 3)], "matches": 2},
    }

    assert [row["book"] for row in rank_candidates(candidates)] == ["Isaiah", "Malachi"]


def test_span_hit_checks_the_annotated_source_range():
    candidate = {"source_intervals": [(36, 40)], "query_intervals": [(1, 3)]}

    assert span_hit(candidate, "whole") is True
    assert span_hit(candidate, "1-2,7-10") is False
    assert span_hit(candidate, "36-40") is True
    assert span_hit(candidate, "40-42") is True       # touches verse 40


def test_query_span_hit_checks_the_quoting_verses():
    candidate = {"source_intervals": [(1, 2)], "query_intervals": [(31, 31)]}

    assert query_span_hit(candidate, "31") is True
    assert query_span_hit(candidate, "30") is False
    assert query_span_hit(candidate, "whole") is True


def test_band_ranking_is_transitive_and_order_independent():
    """A=0.990, B=0.985, C=0.979 with tolerance 0.01.

    Pairwise tolerance is non-transitive here (A~B, B~C, A!~C), which is why the
    ranking groups candidates into anchor bands instead of comparing pairs.
    """
    def candidate(book, best, support):
        return {"book": book, "chapter": 1, "best": best, "support": support,
                "coverage": 0.5, "source_intervals": [(1, 2)],
                "query_intervals": [(1, 2)], "matches": support}

    a = candidate("Isaiah", 0.990, 1)      # weakest support, highest score
    b = candidate("Malachi", 0.985, 5)     # strongest support
    c = candidate("Ezekiel", 0.979, 3)     # outside A's band

    forward = rank_candidates({"i": a, "m": b, "e": c}, tie_tolerance=0.01)
    backward = rank_candidates({"e": dict(c), "m": dict(b), "i": dict(a)}, tie_tolerance=0.01)

    assert [row["book"] for row in forward] == ["Malachi", "Isaiah", "Ezekiel"]
    assert [row["book"] for row in backward] == ["Malachi", "Isaiah", "Ezekiel"]
    assert [row["rank"] for row in forward] == [1, 2, 3]


def test_band_ranking_side_effect_free_across_permutations():
    def candidate(book, best, support):
        return {"book": book, "chapter": 1, "best": best, "support": support,
                "coverage": 0.5, "source_intervals": [(1, 2)],
                "query_intervals": [(1, 2)], "matches": support}

    scores = [("A", 0.991, 1), ("B", 0.989, 4), ("C", 0.980, 2), ("D", 0.960, 9)]
    import itertools

    orders = set()
    for permutation in itertools.permutations(scores):
        candidates = {name: candidate(name, best, support)
                      for name, best, support in permutation}
        ranked = rank_candidates(candidates, tie_tolerance=0.01)
        orders.add(tuple(row["book"] for row in ranked))

    assert len(orders) == 1, f"ranking depends on input order: {orders}"


def test_paired_evidence_records_which_window_matched_which():
    meta = [
        window("Mosiah", 14, 1, 2, corpus="bom"),
        window("Isaiah", 53, 4, 5),
        window("Isaiah", 53, 6, 7),
    ]
    matches = [match(0.9, 0, 1), match(0.7, 0, 2)]

    candidate = aggregate_candidates(meta, matches)[("Isaiah", 53)]

    assert candidate["pairs"][0] == {"sim": 0.9, "query_range": "1-2", "source_range": "4-5"}
    assert candidate["pairs"][1]["source_range"] == "6-7"
    assert candidate["pairs"][0]["sim"] > candidate["pairs"][1]["sim"]


def test_paired_evidence_is_complete_not_truncated():
    """A later classifier must receive every pair, not the strongest ten."""
    meta = [window("Mosiah", 14, 1, 2, corpus="bom")] + [
        window("Isaiah", 53, verse, verse + 1) for verse in range(1, 15)
    ]
    matches = [match(0.99 - 0.01 * index, 0, index)
               for index in range(1, 14)]          # 13 supporting windows

    candidate = aggregate_candidates(meta, matches)[("Isaiah", 53)]

    assert candidate["matches"] == 13
    assert len(candidate["pairs"]) == 13
