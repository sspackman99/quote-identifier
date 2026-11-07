# test_run_matrix.py
"""The matrix driver: one embedding pass, one CSV per pair, one manifest."""

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

import main
import run_matrix

BOM_SAMPLE = """THE FIRST BOOK OF NEPHI HIS REIGN AND MINISTRY (1 Nephi)

1 Nephi Chapter 1

1:1 I, Nephi, having been born of goodly parents, and I did make a record.
1:2 And it came to pass that my father did dwell in a tent.

1 Nephi Chapter 2

2:1 And it came to pass that the Lord spake unto my father.
2:2 And he did keep the commandments of the Lord.
"""

KJV_SAMPLE = """Genesis 1:1\tIn the beginning God created the heaven and the earth.
Genesis 1:2\tAnd the earth was without form and void.
Genesis 2:1\tThus the heavens and the earth were finished.
Genesis 2:2\tAnd on the seventh day God ended his work.
"""


def _vector_for(text):
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vector = rng.rand(8)
    return (vector / np.linalg.norm(vector)).astype("float32")


def _fake_embeddings(windows, text_field="filtered_text", cache=None, **kwargs):
    """Deterministic, cache-aware stand-in for similarity.add_embeddings."""
    texts = [window[text_field] for window in windows]
    compute = lambda batch: np.vstack([_vector_for(text) for text in batch])
    vectors = cache.embed(texts, compute) if cache is not None else compute(texts)
    for window, vector in zip(windows, vectors):
        window["embedding"] = np.asarray(vector, dtype="float32")
    return windows


@pytest.fixture
def corpora(tmp_path, monkeypatch):
    bom = tmp_path / "bom.txt"
    kjv = tmp_path / "kjv.txt"
    bom.write_text(BOM_SAMPLE, encoding="utf-8")
    kjv.write_text(KJV_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(main, "add_embeddings", _fake_embeddings)
    monkeypatch.setattr(main, "identify_ngrams_by_document_frequency",
                        lambda *args, **kwargs: set())
    return {"bom": bom, "kjv": kjv}


def _argv(corpora, output_dir, *pairs, extra=()):
    arguments = []
    for corpus_id, path in corpora.items():
        arguments += ["--corpora", f"{corpus_id}={path}"]
    for query, source in pairs:
        arguments += ["--pair", f"{query}:{source}"]
    arguments += ["--output-dir", str(output_dir), "--no-cache"]
    arguments += list(extra)
    return arguments


def test_each_pair_gets_its_own_csv_and_metadata(corpora, tmp_path):
    out = tmp_path / "pairs"

    status = run_matrix.main(_argv(corpora, out, ("bom", "bom"), ("bom", "kjv")))

    assert status == 0
    for name in ("bom-bom", "bom-kjv"):
        assert (out / f"{name}.csv").exists()
        assert (out / f"{name}.csv.meta.json").exists()


def test_a_cross_pair_keeps_corpus_separation(corpora, tmp_path):
    out = tmp_path / "pairs"
    run_matrix.main(_argv(corpora, out, ("bom", "kjv")))

    frame = pd.read_csv(out / "bom-kjv.csv")

    assert set(frame["corpus_1"]) == {"bom"}
    assert set(frame["corpus_2"]) == {"kjv"}


def test_a_within_pair_stays_inside_one_corpus(corpora, tmp_path):
    out = tmp_path / "pairs"
    run_matrix.main(_argv(corpora, out, ("bom", "bom")))

    frame = pd.read_csv(out / "bom-bom.csv")

    assert (frame["corpus_1"] == "bom").all()
    assert (frame["corpus_2"] == "bom").all()


def test_manifest_records_every_pair_and_the_corpora(corpora, tmp_path):
    out = tmp_path / "pairs"
    run_matrix.main(_argv(corpora, out, ("bom", "bom"), ("bom", "kjv")))

    manifest = json.loads((out / "manifest.json").read_text())

    assert [entry["pair"] for entry in manifest["pairs"]] == ["bom-bom", "bom-kjv"]
    assert [entry["corpus_id"] for entry in manifest["corpora"]] == ["bom", "kjv"]
    assert manifest["top_k"] == run_matrix.TOP_K
    for entry in manifest["pairs"]:
        assert entry["rows"] > 0
        assert entry["max"] <= 1.0
        assert "rows_above_0.80" in entry


def test_the_default_matrix_is_the_ten_approved_pairs():
    assert len(run_matrix.MATRIX) == 10
    assert ("bom", "bom") in run_matrix.MATRIX
    assert ("kjv", "kjv") in run_matrix.MATRIX
    assert ("dc", "pgp") in run_matrix.MATRIX
    assert ("pgp", "dc") not in run_matrix.MATRIX


def test_an_unknown_corpus_is_rejected(corpora, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_matrix.main(_argv(corpora, tmp_path / "pairs", ("bom", "nope")))
    assert exit_info.value.code == 2


def test_a_malformed_pair_is_rejected():
    with pytest.raises(SystemExit):
        run_matrix.parse_pairs(["bom"])


def test_a_malformed_corpus_is_rejected():
    with pytest.raises(SystemExit):
        run_matrix.parse_corpora(["bom"])


def test_the_cache_is_written_and_used(corpora, tmp_path):
    out = tmp_path / "pairs"
    cache_path = tmp_path / "cache.sqlite3"
    argv = _argv(corpora, out, ("bom", "bom"))[:-1]  # drop --no-cache
    argv += ["--cache", str(cache_path)]

    run_matrix.main(argv)

    with sqlite3.connect(cache_path) as connection:
        stored = connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    assert stored > 0, "embeddings must be persisted"

    # a second run must produce the same file and reuse the stored vectors
    first = (out / "bom-bom.csv").read_text()
    run_matrix.main(argv)
    assert (out / "bom-bom.csv").read_text() == first
