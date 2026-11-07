# test_embeddings_cache.py
"""The embedding cache: pay once per distinct text, not once per run."""

import numpy as np
import pytest

from embeddings_cache import EmbeddingCache


def _fake_compute(calls):
    def compute(texts):
        calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype="float32")
    return compute


def test_vectors_come_back_in_input_order(tmp_path):
    calls = []
    with EmbeddingCache(tmp_path / "cache.sqlite3", "m") as cache:
        vectors = cache.embed(["a", "bb", "ccc"], _fake_compute(calls))

    assert vectors.shape == (3, 2)
    assert [row[0] for row in vectors] == [1.0, 2.0, 3.0]
    assert len(calls) == 1


def test_a_repeated_text_is_computed_once(tmp_path):
    calls = []
    with EmbeddingCache(tmp_path / "cache.sqlite3", "m") as cache:
        shared = cache.embed(["same", "same", "other"], _fake_compute(calls))

    assert len(calls[0]) == 2, "the duplicate must not be sent twice"
    assert np.allclose(shared[0], shared[1])


def test_the_cache_survives_a_second_process(tmp_path):
    path = tmp_path / "cache.sqlite3"
    first_calls = []
    with EmbeddingCache(path, "m") as cache:
        original = cache.embed(["a", "b"], _fake_compute(first_calls))
    assert cache.misses == 2

    second_calls = []
    with EmbeddingCache(path, "m") as cache:
        again = cache.embed(["a", "b"], _fake_compute(second_calls))

    assert second_calls == [], "nothing should be recomputed"
    assert cache.hits == 2
    assert np.allclose(original, again)


def test_only_the_missing_texts_are_computed(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with EmbeddingCache(path, "m") as cache:
        cache.embed(["a"], _fake_compute([]))

    calls = []
    with EmbeddingCache(path, "m") as cache:
        cache.embed(["a", "b"], _fake_compute(calls))

    assert calls == [["b"]]
    assert cache.hits == 1 and cache.misses == 1


def test_a_different_model_is_a_different_key(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with EmbeddingCache(path, "model-a") as cache:
        cache.embed(["a"], _fake_compute([]))

    calls = []
    with EmbeddingCache(path, "model-b") as cache:
        cache.embed(["a"], _fake_compute(calls))

    assert calls == [["a"]], "model identity is part of the key"


def test_a_short_batch_is_an_error(tmp_path):
    def bad_compute(texts):
        return np.zeros((len(texts) - 1, 2), dtype="float32")

    with EmbeddingCache(tmp_path / "cache.sqlite3", "m") as cache:
        with pytest.raises(RuntimeError, match="vectors"):
            cache.embed(["a", "b"], bad_compute)


def test_empty_input_returns_an_empty_matrix(tmp_path):
    with EmbeddingCache(tmp_path / "cache.sqlite3", "m") as cache:
        vectors = cache.embed([], _fake_compute([]))

    assert vectors.size == 0
