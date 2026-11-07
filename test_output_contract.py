# test_output_contract.py
"""The CSV contract: pure cosine scoring, top-k per window, one row per pair.

Retrieval runs top-k per source window in both directions; the output collapses
a pair to a single row because cosine similarity is symmetric and the two rows
would otherwise carry the same passages and the same score.
"""

import numpy as np
import pytest

from main import CANDIDATE_COLUMNS, build_candidate_rows, collapse_directions
from top_5_candidates import find_top_5_embedding_candidates


def _windows(angles=(0, 10, 20, 90, 120, 180, 200, 300)):
    """One window per angle on the unit circle, each in its own chapter."""
    windows = []
    for i, angle in enumerate(angles):
        radians = np.radians(angle)
        windows.append({
            "corpus": "bom",
            "book_id": "1-nephi",
            "book": "1 Nephi",
            "chapter": i + 1,
            "start_token_index": 0,
            "start_verse": 1,
            "end_verse": 3,
            "text": f"window {i}",
            "embedding": np.array(
                [np.cos(radians), np.sin(radians)], dtype=np.float32
            ),
        })
    return windows


def _index(windows):
    import faiss

    matrix = np.array([w["embedding"] for w in windows]).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, matrix


def test_score_is_plain_cosine_similarity():
    windows = _windows()
    index, matrix = _index(windows)

    candidates = find_top_5_embedding_candidates(windows, index, matrix)

    assert candidates
    for candidate in candidates:
        source = matrix[candidate["source_window_id"]]
        target = matrix[candidate["candidate_window_id"]]
        expected = float(np.dot(source, target))
        assert candidate["cosine_similarity"] == pytest.approx(expected, abs=1e-5)


def test_ranking_is_descending_cosine_per_source():
    windows = _windows()
    index, matrix = _index(windows)

    candidates = find_top_5_embedding_candidates(windows, index, matrix)

    by_source = {}
    for candidate in candidates:
        by_source.setdefault(candidate["source_window_id"], []).append(candidate)

    for source_id, records in by_source.items():
        scores = [r["cosine_similarity"] for r in records]
        assert scores == sorted(scores, reverse=True), f"source {source_id}"
        assert [r["candidate_rank"] for r in records] == list(
            range(1, len(records) + 1)
        )


def test_every_source_window_gets_top_k_with_no_global_cut():
    windows = _windows()
    index, matrix = _index(windows)
    top_k = 5

    candidates = find_top_5_embedding_candidates(windows, index, matrix, top_k=top_k)

    by_source = {}
    for candidate in candidates:
        by_source.setdefault(candidate["source_window_id"], []).append(candidate)

    assert len(by_source) == len(windows), "every source window must contribute"
    for records in by_source.values():
        assert len(records) == top_k
    assert len(candidates) == len(windows) * top_k


def test_top_k_is_configurable():
    windows = _windows()
    index, matrix = _index(windows)

    candidates = find_top_5_embedding_candidates(windows, index, matrix, top_k=2)

    assert len(candidates) == len(windows) * 2
    assert max(c["candidate_rank"] for c in candidates) == 2


def _candidate(source_id, candidate_id, rank, cosine):
    return {
        "source_window_id": source_id,
        "candidate_window_id": candidate_id,
        "candidate_rank": rank,
        "cosine_similarity": cosine,
    }


def test_collapse_keeps_one_record_per_unordered_pair():
    pairs = collapse_directions([
        _candidate(0, 1, 1, 0.9),
        _candidate(1, 0, 3, 0.9),
    ])

    assert len(pairs) == 1
    assert pairs[0]["pair"] == (0, 1)
    assert pairs[0]["cosine_similarity"] == pytest.approx(0.9)


def test_collapse_keeps_both_sides_ranks():
    """The two directions rank each other differently; neither rank is dropped."""
    pairs = collapse_directions([
        _candidate(0, 1, 1, 0.9),
        _candidate(1, 0, 3, 0.9),
    ])

    assert pairs[0]["ranks"] == {0: 1, 1: 3}


def test_collapse_keeps_a_pair_only_one_side_retrieved():
    """Most pairs are found from one side only; collapsing must not lose them."""
    pairs = collapse_directions([_candidate(5, 9, 2, 0.7)])

    assert len(pairs) == 1
    assert pairs[0]["pair"] == (5, 9)
    assert pairs[0]["ranks"] == {5: 2}


def test_collapse_is_deterministic_and_sorted_by_score():
    pairs = collapse_directions([
        _candidate(0, 1, 1, 0.5),
        _candidate(2, 3, 1, 0.8),
        _candidate(3, 2, 4, 0.8),
    ])

    assert [p["pair"] for p in pairs] == [(2, 3), (0, 1)]


def test_output_rows_carry_both_sides_their_refs_and_their_ranks():
    windows = _windows()
    pairs = collapse_directions([
        _candidate(0, 1, 1, 0.9),
        _candidate(1, 0, 4, 0.9),
    ])

    rows = build_candidate_rows(windows, pairs)

    assert list(rows.columns) == CANDIDATE_COLUMNS
    assert len(rows) == 1
    # No composite-score artefacts survive in the output.
    assert "score" not in rows.columns
    assert "methods" not in rows.columns
    assert "candidate_rank" not in rows.columns

    row = rows.iloc[0]
    assert row["corpus_1"] == "bom" and row["corpus_2"] == "bom"
    assert row["ref_1"] == "1 Nephi 1:1-3"
    assert row["ref_2"] == "1 Nephi 2:1-3"
    assert row["rank_1"] == 1   # window 0 ranked window 1 first
    assert row["rank_2"] == 4   # window 1 ranked window 0 fourth
    assert 0.0 <= row["cosine_similarity"] <= 1.0


def test_one_sided_pair_writes_a_blank_rank(tmp_path):
    """A rank the other side never assigned is empty, not 0 and not 'None'.

    Two rows are written so the rank column holds a mix of present and missing
    values — the case that used to turn the whole column into floats ("1.0").
    """
    windows = _windows()
    pairs = collapse_directions([
        _candidate(0, 1, 1, 0.9),
        _candidate(1, 0, 1, 0.9),
        _candidate(2, 5, 2, 0.5),
    ])
    df = build_candidate_rows(windows, pairs)

    path = tmp_path / "rows.csv"
    df.to_csv(path, index=False)
    header, *lines = path.read_text().splitlines()

    fields = header.split(",")
    rank_1 = fields.index("rank_1")
    rank_2 = fields.index("rank_2")

    two_sided, one_sided = lines[0].split(","), lines[1].split(",")
    assert two_sided[rank_1] == "1" and two_sided[rank_2] == "1"
    assert one_sided[rank_1] == "2" and one_sided[rank_2] == ""


def test_real_retrieval_collapses_without_losing_a_pair():
    windows = _windows()
    index, matrix = _index(windows)

    candidates = find_top_5_embedding_candidates(windows, index, matrix, top_k=3)
    pairs = collapse_directions(candidates)

    directed = {(c["source_window_id"], c["candidate_window_id"]) for c in candidates}
    collapsed = {p["pair"] for p in pairs}

    assert len(collapsed) == len(pairs), "no duplicate pair may survive"
    for a, b in directed:
        assert ((a, b) if a < b else (b, a)) in collapsed
    assert len(pairs) < len(candidates)
