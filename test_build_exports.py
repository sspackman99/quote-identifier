# test_build_exports.py
"""The export step: references parsed, columns stable, self-pairs counted once."""

import json
from pathlib import Path

import pandas as pd

import build_exports as bx
from build_exports import (
    CSV_COLUMNS,
    build_csv,
    build_span_texts,
    build_viz_data,
    viz_corpus,
    parse_ref,
    parse_span,
    to_frame,
)


def _frame(rows):
    base = {
        "pair": "bom-bom", "cosine_similarity": 0.9,
        "query_corpus": "bom", "query_ref": "2 Nephi 6:6", "query_span": "6-6",
        "source_corpus": "bom", "source_ref": "1 Nephi 21:22", "source_span": "22-22",
        "relationship": "quotation", "secondary_relationship": "",
        "confidence": 0.95, "needs_review": False, "review_reasons": "",
        "reproduced_share": 1.0, "extent_reading": "quotation",
        "classifier_query_span": "6", "classifier_source_span": "22",
        "guard_applied": "", "explanation": "reproduced", "evidence": "a phrase",
        "model": "m", "rubric_version": "1.0", "error": "",
    }
    return to_frame([{**base, **row} for row in rows])


def test_references_parse_into_book_and_chapter():
    assert parse_ref("1 Nephi 21:22") == ("1 Nephi", 21)
    assert parse_ref("D&C 76:22") == ("D&C", 76)
    assert parse_ref("Joseph Smith-History 1:37") == ("Joseph Smith-History", 1)
    assert parse_ref("nonsense") == (None, None)


def test_spans_parse_into_ranges():
    assert parse_span("22-24") == (22, 24)
    assert parse_span("22") == (22, 22)
    assert parse_span("") == (None, None)
    assert parse_span("nan") == (None, None)


def test_csv_has_the_declared_columns_and_sorts_by_score(tmp_path):
    frame = _frame([
        {"cosine_similarity": 0.81},
        {"cosine_similarity": 0.99},
    ])
    path = tmp_path / "out.csv"
    build_csv(frame, path)

    written = pd.read_csv(path)
    assert list(written.columns) == CSV_COLUMNS
    assert written.iloc[0]["cosine_similarity"] == 0.99


def test_viz_counts_a_self_pair_once_per_node(tmp_path):
    frame = _frame([{
        "query_ref": "Jacob 5:20", "query_span": "20-20",
        "source_ref": "Jacob 5:30-32", "source_span": "30-32",
    }])
    path = tmp_path / "data.json"
    payload = build_viz_data(frame, path, texts={})

    assert len(payload["nodes"]) == 1
    assert payload["nodes"][0]["pairs"] == 1, "a book must not be double counted"
    assert payload["edges"][0]["source"] == payload["edges"][0]["target"]


def test_viz_edges_aggregate_both_directions(tmp_path):
    frame = _frame([
        {"query_ref": "2 Nephi 6:6", "query_span": "6-6",
         "source_ref": "Isaiah 49:6", "source_span": "6-6",
         "query_corpus": "bom", "source_corpus": "kjv"},
        {"query_ref": "Isaiah 49:6", "query_span": "6-6",
         "source_ref": "2 Nephi 6:6", "source_span": "6-6",
         "query_corpus": "kjv", "source_corpus": "bom"},
    ])
    path = tmp_path / "data.json"
    payload = build_viz_data(frame, path, texts={})

    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["count"] == 2
    assert payload["totals"]["pairs"] == 2


def test_viz_data_is_written_as_js_too(tmp_path):
    frame = _frame([{}])
    path = tmp_path / "data.json"
    build_viz_data(frame, path)

    js = (tmp_path / "data.js").read_text()
    assert js.startswith("window.VIZ_DATA = ")
    assert json.loads(js[len("window.VIZ_DATA = "):].rstrip().rstrip(";"))["totals"]["pairs"] == 1


def test_span_texts_are_built_from_the_corpora():
    texts = {("bom", "Jacob", 5, 20): "And the servant said unto his lord.",
             ("bom", "Jacob", 5, 30): "And it came to pass that the Lord said."}
    frame = _frame([{
        "query_ref": "Jacob 5:20", "query_span": "20-20",
        "source_ref": "Jacob 5:30-30", "source_span": "30-30",
    }])

    spans = build_span_texts(frame, texts)
    payload = build_viz_data(frame, Path("/tmp/viz_test.json"), texts=texts)

    assert "5:20 And the servant said unto his lord." in spans["bom|Jacob|5|20-20"]
    assert payload["pairs"][0]["qk"] == "bom|Jacob|5|20-20"
    assert payload["spans"][payload["pairs"][0]["sk"]].startswith("5:30 And it came to pass")


def test_testaments_are_named_for_bible_books():
    assert bx.testament_of("kjv", "Isaiah") == "OT"
    assert bx.testament_of("kjv", "Revelation") == "NT"
    assert bx.testament_of("bom", "1 Nephi") == ""
    assert viz_corpus("kjv", "Genesis") == "ot"
    assert viz_corpus("kjv", "Matthew") == "nt"
    assert viz_corpus("bom", "1 Nephi") == "bom"


def test_csv_carries_the_testament(tmp_path):
    frame = _frame([{
        "query_corpus": "kjv", "query_ref": "Matthew 7:5", "query_span": "5-5",
        "source_corpus": "kjv", "source_ref": "Luke 6:42", "source_span": "42-42",
    }])
    path = tmp_path / "out.csv"
    build_csv(frame, path)

    row = pd.read_csv(path).iloc[0]
    assert row.query_testament == "NT"
    assert row.source_testament == "NT"


def test_viz_groups_bible_books_by_testament(tmp_path):
    frame = _frame([{
        "query_corpus": "kjv", "query_ref": "Matthew 7:5", "query_span": "5-5",
        "source_corpus": "kjv", "source_ref": "Isaiah 49:6", "source_span": "6-6",
    }])
    payload = build_viz_data(frame, tmp_path / "data.json", texts={})

    corpora = {n["book"]: n["corpus"] for n in payload["nodes"]}
    assert corpora["Matthew"] == "nt"
    assert corpora["Isaiah"] == "ot"
    assert "kjv" not in payload["corpora"]
