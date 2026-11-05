# test_embed_http.py
"""Tests for the remote embeddings path in similarity.py.

These exercise the HTTP provider without a GPU, torch, or a live server: the
endpoint is faked at the urllib layer.
"""

import json

import numpy as np
import pytest

from similarity import _normalize_rows, embed_texts_via_http


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeEndpoint:
    """Records requests and returns deterministic, deliberately unnormalized vectors."""

    def __init__(self, shuffle=False, drop=False):
        self.requests = []
        self.shuffle = shuffle
        self.drop = drop

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        assert request.headers.get("Content-type") == "application/json"
        self.requests.append(body)

        data = []
        for index, text in enumerate(body["input"]):
            # length-dependent vector so ordering errors are visible
            base = float(len(text))
            data.append({"index": index, "object": "embedding",
                         "embedding": [base, base + 1.0, base + 2.0]})

        if self.drop:
            data = data[:-1]
        if self.shuffle:
            data = list(reversed(data))

        return FakeResponse({"model": body["model"], "object": "list",
                             "usage": {"prompt_tokens": 0, "total_tokens": 0},
                             "data": data})


def test_embed_texts_via_http_normalizes_rows(monkeypatch):
    endpoint = FakeEndpoint()
    monkeypatch.setattr("similarity.urllib.request.urlopen", endpoint)

    matrix = embed_texts_via_http(["a", "bb", "ccc"], "http://host:8001/v1/embeddings",
                                  "embeddinggemma-300m", batch_size=64)

    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # first row was [1,2,3] before normalisation
    assert np.allclose(matrix[0], np.array([1, 2, 3], dtype=np.float32) / np.sqrt(14))


def test_embed_texts_via_http_keeps_input_order_when_endpoint_shuffles(monkeypatch):
    endpoint = FakeEndpoint(shuffle=True)
    monkeypatch.setattr("similarity.urllib.request.urlopen", endpoint)

    matrix = embed_texts_via_http(["a", "bb", "ccc"], "http://host:8001/v1/embeddings",
                                  "embeddinggemma-300m")

    # rows must follow the INPUT order, not the endpoint's response order
    lengths = np.linalg.norm(matrix, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-6)
    assert matrix[0, 0] < matrix[1, 0] < matrix[2, 0]


def test_embed_texts_via_http_batches_requests(monkeypatch):
    endpoint = FakeEndpoint()
    monkeypatch.setattr("similarity.urllib.request.urlopen", endpoint)

    texts = [f"text-{i}" for i in range(5)]
    matrix = embed_texts_via_http(texts, "http://host:8001/v1/embeddings",
                                  "embeddinggemma-300m", batch_size=2)

    assert matrix.shape == (5, 3)
    assert [len(call["input"]) for call in endpoint.requests] == [2, 2, 1]
    assert all(call["model"] == "embeddinggemma-300m" for call in endpoint.requests)


def test_embed_texts_via_http_rejects_short_response(monkeypatch):
    endpoint = FakeEndpoint(drop=True)
    monkeypatch.setattr("similarity.urllib.request.urlopen", endpoint)

    with pytest.raises(RuntimeError, match="vectors for"):
        embed_texts_via_http(["a", "bb"], "http://host:8001/v1/embeddings",
                             "embeddinggemma-300m")


def test_add_embeddings_uses_env_endpoint(monkeypatch):
    from similarity import add_embeddings

    endpoint = FakeEndpoint()
    monkeypatch.setattr("similarity.urllib.request.urlopen", endpoint)
    monkeypatch.setenv("QI_EMBEDDINGS_URL", "http://host:8001/v1/embeddings")
    monkeypatch.setenv("QI_EMBEDDINGS_MODEL", "embeddinggemma-300m")

    windows = [{"filtered_text": "alpha"}, {"filtered_text": "beta beta"}]
    out = add_embeddings(windows)

    assert len(out) == 2
    for window in out:
        assert window["embedding"].shape == (3,)
        assert abs(float(np.linalg.norm(window["embedding"])) - 1.0) < 1e-6
    assert endpoint.requests[0]["model"] == "embeddinggemma-300m"


def test_normalize_rows_handles_zero_vector():
    matrix = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
    out = _normalize_rows(matrix)
    assert np.allclose(out[0], [0.0, 0.0])
    assert np.allclose(out[1], [0.6, 0.8])
