# test_classify_matrix.py
"""The overnight driver: walk each pair CSV with the right corpus roles."""

import json

import pandas as pd
import pytest

import classify_matrix
import classify_pairs

PAIR_COLUMNS = [
    "cosine_similarity", "rank_1", "rank_2",
    "corpus_1", "book_id_1", "book_1", "chapter_1", "verse_1", "end_verse_1",
    "ref_1", "text_1",
    "corpus_2", "book_id_2", "book_2", "chapter_2", "verse_2", "end_verse_2",
    "ref_2", "text_2",
]

ANSWER = {
    "relationship": "quotation",
    "confidence": 0.9,
    "query_span": "6",
    "source_span": "22",
    "secondary_relationship": "",
    "explanation": "reproduced",
    "evidence": ["mine hand"],
    "needs_review": False,
}


def _row(cosine, side_1, side_2, corpus_1="bom", corpus_2="bom"):
    book_1, chapter_1, verse_1 = side_1
    book_2, chapter_2, verse_2 = side_2
    return {
        "cosine_similarity": cosine, "rank_1": 1, "rank_2": 1,
        "corpus_1": corpus_1, "book_id_1": book_1.lower(), "book_1": book_1,
        "chapter_1": chapter_1, "verse_1": verse_1, "end_verse_1": verse_1,
        "ref_1": f"{book_1} {chapter_1}:{verse_1}", "text_1": "a",
        "corpus_2": corpus_2, "book_id_2": book_2.lower(), "book_2": book_2,
        "chapter_2": chapter_2, "verse_2": verse_2, "end_verse_2": verse_2,
        "ref_2": f"{book_2} {chapter_2}:{verse_2}", "text_2": "b",
    }


@pytest.fixture
def pair_files(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    pd.DataFrame([
        _row(0.99, ("1 Nephi", 21, 22), ("2 Nephi", 6, 6)),
    ], columns=PAIR_COLUMNS).to_csv(pairs / "bom-bom.csv", index=False)
    pd.DataFrame([
        _row(0.99, ("Mosiah", 14, 1), ("Isaiah", 53, 1), "bom", "kjv"),
    ], columns=PAIR_COLUMNS).to_csv(pairs / "bom-kjv.csv", index=False)

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            return json.dumps(ANSWER)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)
    return pairs


def test_only_requested_pairs_are_classified(pair_files, tmp_path):
    out = tmp_path / "classifications"

    status = classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "bom-kjv",
    ])

    assert status == 0
    assert (out / "bom-kjv.csv").exists()
    assert (out / "bom-kjv.jsonl").exists()
    assert not (out / "bom-bom.csv").exists()


def test_a_cross_pair_is_oriented_with_the_source_corpus(pair_files, tmp_path):
    out = tmp_path / "classifications"

    classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "bom-kjv",
    ])

    frame = pd.read_csv(out / "bom-kjv.csv")
    assert frame.iloc[0]["query_corpus"] == "bom"
    assert frame.iloc[0]["source_corpus"] == "kjv"
    assert frame.iloc[0]["source_ref"] == "Isaiah 53:1"


def test_a_within_pair_uses_canonical_order(pair_files, tmp_path):
    out = tmp_path / "classifications"

    classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "bom-bom",
    ])

    frame = pd.read_csv(out / "bom-bom.csv")
    assert frame.iloc[0]["query_ref"] == "2 Nephi 6:6"
    assert frame.iloc[0]["source_ref"] == "1 Nephi 21:22"


def test_missing_pair_files_are_skipped_not_fatal(pair_files, tmp_path):
    out = tmp_path / "classifications"

    status = classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "dc-pgp",
    ])

    assert status == 0
    summary = json.loads((out / "matrix_classification.json").read_text())
    assert summary["pairs"] == []


def test_the_cutoff_defaults_differ_by_pair_type(pair_files, tmp_path):
    out = tmp_path / "classifications"
    classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "bom-bom", "--pairs", "bom-kjv",
    ])

    summary = json.loads((out / "matrix_classification.json").read_text())
    cutoffs = {entry["pair"]: entry["cutoff"] for entry in summary["pairs"]}

    assert cutoffs["bom-bom"] == classify_matrix.DEFAULT_WITHIN_CUTOFF
    assert cutoffs["bom-kjv"] == classify_matrix.DEFAULT_CROSS_CUTOFF
    assert cutoffs["bom-bom"] > cutoffs["bom-kjv"]


def test_the_priority_order_starts_with_cross_corpus_pairs():
    assert classify_matrix.PRIORITY[0] == ("bom", "kjv")
    assert classify_matrix.PRIORITY[-1] == ("kjv", "kjv")


def test_a_per_pair_cutoff_overrides_the_type_default(pair_files, tmp_path):
    out = tmp_path / "classifications"
    classify_matrix.main([
        "--input-dir", str(pair_files), "--output-dir", str(out),
        "--pairs", "bom-bom", "--pairs", "bom-kjv",
        "--pair-cutoff", "bom-kjv=0.93",
    ])

    summary = json.loads((out / "matrix_classification.json").read_text())
    cutoffs = {entry["pair"]: entry["cutoff"] for entry in summary["pairs"]}

    assert cutoffs["bom-kjv"] == 0.93
    assert cutoffs["bom-bom"] == classify_matrix.DEFAULT_WITHIN_CUTOFF


def test_a_malformed_pair_cutoff_is_rejected(pair_files, tmp_path):
    with pytest.raises(SystemExit):
        classify_matrix.main([
            "--input-dir", str(pair_files), "--output-dir", str(tmp_path / "c"),
            "--pair-cutoff", "kjv-kjv",
        ])
