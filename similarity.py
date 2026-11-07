# similarity.py

import json
import os
import urllib.request

import numpy as np
import faiss
from datasketch import MinHash, MinHashLSH

DEFAULT_LOCAL_MODEL = "all-mpnet-base-v2"
DEFAULT_HTTP_MODEL = "embeddinggemma-300m"
EMBEDDINGS_URL_ENV = "QI_EMBEDDINGS_URL"
EMBEDDINGS_MODEL_ENV = "QI_EMBEDDINGS_MODEL"

# -------------------------------------------------
# 1. MINHASH
# -------------------------------------------------

def compute_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf8"))
    return m


def add_minhashes(windows, text_field="filtered_text"):
    for window in windows:
        window["minhash"] = compute_minhash(window[text_field])
    return windows


def build_lsh(windows, threshold=0.6, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, window in enumerate(windows):
        lsh.insert(str(i), window["minhash"])
    return lsh


def find_minhash_matches(windows, lsh):
    matches = []
    for i, window in enumerate(windows):
        result = lsh.query(window["minhash"])
        for r in result:
            j = int(r)
            if j != i:
                matches.append((i, j, "minhash"))
    return matches

# -------------------------------------------------
# 2. EMBEDDINGS + FAISS (GPU ENABLED)
# -------------------------------------------------

def _normalize_rows(matrix):
    """L2-normalize each row so cosine similarity is a plain inner product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def embed_texts_via_http(texts, url, model, batch_size=64, timeout=300):
    """Embed texts through an OpenAI-compatible POST /v1/embeddings endpoint.

    Lets a machine without a GPU — or without torch/sentence-transformers at all
    — use an embedding server running on a host that has one, e.g. the llama.cpp
    server on ozai: QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings.

    Returns an L2-normalized float32 array, one row per input text.
    """
    batch_arrays = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))

        items = body.get("data") or []
        if not items:
            raise RuntimeError(
                f"embedding endpoint {url} returned no data: {str(body)[:200]}"
            )
        if all("index" in item for item in items):
            items = sorted(items, key=lambda item: item["index"])
        if len(items) != len(batch):
            raise RuntimeError(
                f"embedding endpoint returned {len(items)} vectors "
                f"for {len(batch)} inputs"
            )

        # Convert each batch to float32 immediately: accumulating the JSON
        # response as Python floats costs ~8x the memory of the array, which
        # matters once a corpus reaches tens of thousands of windows.
        batch_arrays.append(
            np.asarray([item["embedding"] for item in items], dtype="float32")
        )
        print(f"  embedded {min(start + batch_size, len(texts))}/{len(texts)} windows",
              flush=True)

    return _normalize_rows(np.vstack(batch_arrays))


def embed_texts(texts,
                model_name=None,
                batch_size=64,
                url=None):
    """Embed a list of strings, L2-normalized, one row per input.

    Over HTTP when an endpoint is configured (the `url` argument or
    QI_EMBEDDINGS_URL), in which case torch / sentence-transformers are never
    imported — that is what lets this pipeline run on a CPU-only box using a GPU
    host's embedding server. Otherwise the model runs locally.
    """
    url = url or os.environ.get(EMBEDDINGS_URL_ENV)

    if url:
        model = model_name or os.environ.get(EMBEDDINGS_MODEL_ENV, DEFAULT_HTTP_MODEL)
        print(f"Embedding via {url} (model {model})", flush=True)
        return embed_texts_via_http(texts, url, model, batch_size=batch_size)

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name or DEFAULT_LOCAL_MODEL, device=device)

    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # IMPORTANT
    )


def add_embeddings(windows,
                   text_field="filtered_text",
                   model_name=None,
                   batch_size=64,
                   url=None,
                   cache=None):
    """Attach an L2-normalized embedding to every window.

    `cache` is anything with `.embed(texts, compute)` — see embeddings_cache —
    which lets a corpus be embedded once instead of once per run.
    """
    texts = [w[text_field] for w in windows]

    if cache is not None:
        embeddings = cache.embed(
            texts,
            lambda missing: embed_texts(missing, model_name=model_name,
                                        batch_size=batch_size, url=url),
        )
    else:
        embeddings = embed_texts(texts, model_name=model_name,
                                 batch_size=batch_size, url=url)

    for window, emb in zip(windows, embeddings):
        window["embedding"] = emb

    return windows


def build_faiss_index(windows):
    dim = len(windows[0]["embedding"])
    
    # Use inner product because embeddings are normalized
    index = faiss.IndexFlatIP(dim)
    
    embeddings = np.array([w["embedding"] for w in windows]).astype("float32")
    index.add(embeddings)
    
    return index, embeddings


def find_embedding_matches(windows, index, embeddings, k=5):
    matches = []
    for i, emb in enumerate(embeddings):
        D, I = index.search(np.array([emb]), k)
        for j in I[0]:
            if j != i:
                matches.append((i, j, "embedding"))
    return matches

# -------------------------------------------------
# 3. PAIR MAINTENANCE
# -------------------------------------------------

def merge_duplicate_matches(matches):
    """Collapse candidate triples into one record per unordered pair.

    The embedding and MinHash pipelines each report a pair, and each reports it
    in both directions (i->j and j->i), so the same pair can arrive four times.
    Each surviving record keeps every method that found it — no information is
    lost — and the pair is ordered (min, max) so the output is stable.
    """
    merged = {}

    for i, j, method in matches:
        if i == j:
            continue
        key = (i, j) if i < j else (j, i)
        merged.setdefault(key, set()).add(method)

    return [(i, j, sorted(methods)) for (i, j), methods in sorted(merged.items())]


def filter_nearby_matches(windows, matches, min_token_distance=200):
    filtered = []
    for i, j, source in matches:
        w1 = windows[i]
        w2 = windows[j]
        
        # Skip if same chapter and too close
        if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
            distance = abs(w1["start_token_index"] - w2["start_token_index"])
            if distance < min_token_distance:
                continue
        
        filtered.append((i, j, source))
    
    return filtered