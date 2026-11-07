# test_dedupe.py
"""Tests for collapsing duplicate candidate pairs.

Both retrieval pipelines report a pair, and each reports it in both directions,
so the same pair used to appear up to four times in the output.
"""

import numpy as np

from similarity import (
    filter_nearby_matches,
    merge_duplicate_matches,
)


def test_merge_collapses_directions_and_methods():
    matches = [
        (0, 5, "embedding"),
        (5, 0, "embedding"),
        (0, 5, "minhash"),
        (5, 0, "minhash"),
    ]

    assert merge_duplicate_matches(matches) == [(0, 5, ["embedding", "minhash"])]


def test_merge_keeps_pairs_found_by_one_method():
    matches = [(0, 5, "embedding"), (1, 2, "minhash")]

    assert merge_duplicate_matches(matches) == [
        (0, 5, ["embedding"]),
        (1, 2, ["minhash"]),
    ]


def test_merge_drops_self_matches_and_is_deterministic():
    matches = [(3, 3, "embedding"), (7, 2, "minhash"), (2, 7, "embedding")]

    first = merge_duplicate_matches(matches)
    second = merge_duplicate_matches(list(matches))

    assert first == second
    assert first == [(2, 7, ["embedding", "minhash"])]


def _window(book, chapter, token_index):
    return {
        "book": book,
        "chapter": chapter,
        "start_verse": "1",
        "end_verse": "1",
        "start_token_index": token_index,
        "tokens": ["token"] * 5,
        "filtered_tokens": ["token"] * 5,
        "text": f"{book} {chapter} {token_index}",
        "embedding": np.array([1.0, 0.0], dtype=np.float32),
        "minhash": None,
    }


def test_dedupe_then_filter_keeps_one_record_per_pair():
    windows = [
        _window("Genesis", 1, 0),
        _window("Exodus", 1, 1000),
    ]
    raw = [
        (0, 1, "embedding"),
        (1, 0, "embedding"),
        (0, 1, "minhash"),
        (1, 0, "minhash"),
    ]

    merged = merge_duplicate_matches(raw)
    filtered = filter_nearby_matches(windows, merged, min_token_distance=200)

    assert len(filtered) == 1
    i, j, methods = filtered[0]
    assert (i, j) == (0, 1)
    assert methods == ["embedding", "minhash"]
