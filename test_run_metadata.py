# test_run_metadata.py
"""Every generated CSV must be traceable to the settings that produced it."""

import json
from pathlib import Path

from main import write_run_metadata


def test_metadata_sidecar_records_the_run_settings(tmp_path):
    output = tmp_path / "matches.csv"
    output.write_text("score,methods\n")

    meta_path = write_run_metadata(
        output,
        corpus_id="bom",
        corpus_path="data/bom.txt",
        corpus_digest="abc123",
        filter_metric="document",
        filter_threshold=0.2,
        window_size=30,
        window_overlap=15,
        rows=750,
    )

    assert meta_path == Path(str(output) + ".meta.json")
    recorded = json.loads(meta_path.read_text())

    assert recorded["corpus_id"] == "bom"
    assert recorded["corpus_digest"] == "abc123"
    assert recorded["filter_metric"] == "document"
    assert recorded["filter_threshold"] == 0.2
    assert recorded["window_size"] == 30
    assert recorded["window_overlap"] == 15
    assert recorded["rows"] == 750
    assert recorded["generated_at_utc"].endswith("+00:00")
    assert "git_commit" in recorded


def test_metadata_is_json_serialisable_for_nested_values(tmp_path):
    output = tmp_path / "matches.csv"

    meta_path = write_run_metadata(
        output,
        corpus_id="kjv",
        extra={"embedding_endpoint": "http://host:8001/v1/embeddings"},
    )

    recorded = json.loads(meta_path.read_text())
    assert recorded["extra"]["embedding_endpoint"].endswith("/v1/embeddings")
