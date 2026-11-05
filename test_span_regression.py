# test_span_regression.py
"""Regression tests for span-aware retrieval on the four known hard cases.

These need the local index cache (`.eval_cache/`), which is a derived artifact and
not in git, so the module skips when it is missing. Run
`python evaluate_retrieval.py --cache-index .eval_cache` once to build it.
"""

import json
from pathlib import Path

import pytest

from candidates import (
    aggregate_candidates,
    collect_matches,
    rank_candidates,
    span_hit,
    verses_per_chapter,
)
from evaluate_retrieval import parse_ref

ROOT = Path(__file__).parent
CACHE = ROOT / ".eval_cache"


def _require(name, path):
    if not path.exists():
        pytest.skip(f"no {name} at {path}")


@pytest.fixture(scope="module")
def bundle():
    for name, path in (
        ("book of mormon", ROOT / "data" / "bom.txt"),
        ("king james text", ROOT / "data" / "kjv.txt"),
        ("index cache", CACHE / "embeddings.npy"),
        ("index metadata", CACHE / "windows.json"),
    ):
        _require(name, path)

    import faiss
    import numpy as np

    meta = json.loads((CACHE / "windows.json").read_text())
    embeddings = np.load(CACHE / "embeddings.npy")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "meta": meta,
        "embeddings": embeddings,
        "index": index,
        "sizes": verses_per_chapter(meta),
    }


def evaluate(bundle, query, expected, neighbors=50, tolerance=0.01):
    """Rank candidates for `query` and return the expected chapter's evidence."""
    meta, embeddings, index = bundle["meta"], bundle["embeddings"], bundle["index"]

    query_book, query_chapter = parse_ref(query)
    query_indices = [
        i for i, window in enumerate(meta)
        if window["corpus"] == "bom"
        and window["book"] == query_book
        and window["chapter"] == query_chapter
    ]
    assert query_indices, f"no windows for {query}"

    matches = collect_matches(meta, index, embeddings, query_indices, neighbors)
    candidates = aggregate_candidates(meta, matches, bundle["sizes"])
    ranked = rank_candidates(candidates, tie_tolerance=tolerance)

    expected_book, expected_chapter = parse_ref(expected)
    candidate = candidates[(expected_book, expected_chapter)]
    top = ranked[0]

    return {
        "candidate": candidate,
        "span_rank": candidate["rank"],
        "best": candidate["best"],
        "support": candidate["support"],
        "source_range": candidate["source_range"],
        "query_range": candidate["query_range"],
        "top1": (top["book"], top["chapter"]),
    }


def test_moroni_10_single_verse_quotation_is_recovered(bundle):
    """Moroni 10:31 quotes Isaiah 52:1-2; one window must not be drowned out.

    Under chapter-hit ranking this pair sat at rank 30.
    """
    result = evaluate(bundle, "Moroni 10", "Isaiah 52")

    assert result["span_rank"] <= 2, result
    assert span_hit(result["candidate"], "1-2") is True, result["source_range"]
    assert result["best"] > 0.75
    assert result["query_range"], "the quoting verse range should be reported"


def test_mosiah_12_quoted_block_ranks_high(bundle):
    """Mosiah 12:21-24 reproduces Isaiah 52:7-10."""
    result = evaluate(bundle, "Mosiah 12", "Isaiah 52")

    assert result["span_rank"] <= 2, result
    assert result["best"] > 0.90


def test_three_nephi_20_quotation_still_ranks_first(bundle):
    """3 Nephi 20:36-40 quotes Isaiah 52; span ranking must not displace it."""
    result = evaluate(bundle, "3 Nephi 20", "Isaiah 52")

    assert result["span_rank"] == 1, result


def test_mosiah_14_full_chapter_quotation_is_unchanged(bundle):
    """The whole-chapter quotation must still rank first after the change."""
    result = evaluate(bundle, "Mosiah 14", "Isaiah 53")

    assert result["span_rank"] == 1, result
    assert result["best"] > 0.95


def test_candidate_evidence_is_reported_for_inspection(bundle):
    result = evaluate(bundle, "Mosiah 14", "Isaiah 53")

    assert result["support"] >= 1
    assert result["source_range"]
    assert "-" in result["source_range"] or result["source_range"].isdigit()
