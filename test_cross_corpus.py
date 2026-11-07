# test_cross_corpus.py
"""Pooled retrieval: one index over two corpora, and the --pairs switch."""

import numpy as np
import pandas as pd
import pytest

import main

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


def _window(corpus, book, chapter, token_index):
    return {
        "corpus": corpus,
        "book": book,
        "chapter": chapter,
        "start_token_index": token_index,
    }


def test_select_pair_mode_keeps_the_requested_side():
    windows = [
        _window("bom", "1 Nephi", 1, 0),
        _window("bom", "1 Nephi", 2, 0),
        _window("kjv", "Genesis", 1, 0),
        _window("kjv", "Genesis", 2, 0),
    ]
    pairs = [
        {"pair": (0, 1), "cosine_similarity": 0.9},   # within bom
        {"pair": (0, 2), "cosine_similarity": 0.8},   # cross
        {"pair": (2, 3), "cosine_similarity": 0.7},   # within kjv
        {"pair": (1, 3), "cosine_similarity": 0.6},   # cross
    ]

    assert len(main.select_pair_mode(pairs, windows, "both")) == 4
    assert [p["pair"] for p in main.select_pair_mode(pairs, windows, "cross")] == [(0, 2), (1, 3)]
    assert [p["pair"] for p in main.select_pair_mode(pairs, windows, "within")] == [(0, 1), (2, 3)]


@pytest.fixture
def corpora(tmp_path):
    bom = tmp_path / "bom.txt"
    kjv = tmp_path / "kjv.txt"
    bom.write_text(BOM_SAMPLE, encoding="utf-8")
    kjv.write_text(KJV_SAMPLE, encoding="utf-8")
    return bom, kjv


def _fake_embeddings(windows, **kwargs):
    """Deterministic unit vectors, so the test needs no GPU and no server."""
    for position, window in enumerate(windows):
        rng = np.random.RandomState(position + 1)
        vector = rng.rand(8)
        window["embedding"] = (vector / np.linalg.norm(vector)).astype("float32")
    return windows


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(main, "add_embeddings", _fake_embeddings)
    # tiny corpora have no room for a document-frequency bar; keep the text
    monkeypatch.setattr(main, "identify_ngrams_by_document_frequency",
                        lambda *args, **kwargs: set())


def test_common_phrases_are_judged_per_corpus_not_pooled():
    """A phrase common in one canon must be filtered even if rare when pooled."""
    windows = []
    for chapter in range(6):
        tokens = (["and", "it", "came", "to", "pass"] if chapter < 2
                  else [f"a{chapter}"] * 5)
        windows.append({"corpus": "bom", "book": "1 Nephi",
                        "chapter": chapter, "tokens": tokens})
    for chapter in range(6):
        windows.append({"corpus": "kjv", "book": "Genesis",
                        "chapter": chapter, "tokens": [f"b{chapter}"] * 5})

    phrases, per_corpus = main.common_phrases(windows, ["bom", "kjv"])

    # 2 of 6 chapters is 33%, over the 20% bar; pooled it is 2 of 12 (17%) and
    # would have slipped through.
    assert ("and", "it", "came") in phrases
    assert per_corpus["bom"][1] == 6, "documents counted per corpus"
    assert per_corpus["bom"][0] >= 1
    assert per_corpus["kjv"][1] == 6


def test_pooled_run_writes_only_cross_corpus_pairs(corpora, stubbed, tmp_path):
    bom, kjv = corpora
    df = main.run(bom, tmp_path / "pairs.csv", corpus_id="bom",
                  corpus_b=kjv, corpus_id_b="kjv", pair_mode="cross")

    assert len(df) > 0
    assert set(df["corpus_1"]) == {"bom"}
    assert set(df["corpus_2"]) == {"kjv"}
    assert list(df.columns) == main.CANDIDATE_COLUMNS


def test_pooled_run_can_keep_within_corpus_pairs(corpora, stubbed, tmp_path):
    bom, kjv = corpora
    df = main.run(bom, tmp_path / "pairs.csv", corpus_id="bom",
                  corpus_b=kjv, corpus_id_b="kjv", pair_mode="within")

    assert len(df) > 0
    assert (df["corpus_1"] == df["corpus_2"]).all()


def test_pooled_run_records_both_corpora_in_the_metadata(corpora, stubbed, tmp_path):
    import json

    bom, kjv = corpora
    output = tmp_path / "pairs.csv"
    main.run(bom, output, corpus_id="bom",
             corpus_b=kjv, corpus_id_b="kjv", pair_mode="cross")

    meta = json.loads((tmp_path / "pairs.csv.meta.json").read_text())
    assert meta["pair_mode"] == "cross"
    assert [entry["corpus_id"] for entry in meta["corpora"]] == ["bom", "kjv"]
    assert meta["corpora"][1]["verses"] == 4
    assert meta["windows"] == sum(entry["windows"] for entry in meta["corpora"])


def test_single_corpus_run_is_unchanged(corpora, stubbed, tmp_path):
    bom, _ = corpora
    df = main.run(bom, tmp_path / "pairs.csv", corpus_id="bom")

    assert len(df) > 0
    assert set(df["corpus_1"]) == {"bom"}
    assert set(df["corpus_2"]) == {"bom"}


def test_cli_refuses_cross_without_a_second_corpus(corpora, capsys):
    bom, _ = corpora
    with pytest.raises(SystemExit) as exit_info:
        main.main(["--corpus", str(bom), "--pairs", "cross"])
    assert exit_info.value.code == 2


def test_cli_refuses_the_same_file_twice(corpora):
    bom, _ = corpora
    with pytest.raises(SystemExit) as exit_info:
        main.main(["--corpus", str(bom), "--corpus-b", str(bom), "--pairs", "cross"])
    assert exit_info.value.code == 2


def test_pair_csv_carries_a_corpus_column_on_both_sides(corpora, stubbed, tmp_path):
    bom, kjv = corpora
    output = tmp_path / "pairs.csv"
    main.run(bom, output, corpus_id="bom", corpus_b=kjv, corpus_id_b="kjv",
             pair_mode="cross")

    frame = pd.read_csv(output)
    assert {"corpus_1", "corpus_2"} <= set(frame.columns)
    assert frame["ref_1"].str.len().gt(0).all()
    assert frame["ref_2"].str.len().gt(0).all()
