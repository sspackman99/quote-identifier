"""Content-addressed embedding cache, so a passage is never paid for twice.

The matrix runs the same corpora through several pairs (a Book of Mormon window is
embedded for bom-bom, bom-kjv, dc-bom and pgp-bom). Embedding is the only part of
retrieval that costs GPU time, so it is keyed by the text itself plus the model
name: run ten pairs and each distinct passage is embedded once.

Storage is SQLite with a float32 BLOB per vector. 80,000 windows of 768
dimensions is about 240 MB, which is cheaper than re-embedding and survives a
killed run.
"""

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
)
"""


class EmbeddingCache:
    """Look up embeddings by content; compute and store whatever is missing."""

    def __init__(self, path, model):
        self.path = Path(path)
        self.model = model
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.execute(SCHEMA)
        self.connection.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(model, text):
        return hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()

    def _lookup(self, keys):
        found = {}
        chunk = 800  # stay under SQLite's variable limit
        for start in range(0, len(keys), chunk):
            batch = keys[start:start + chunk]
            placeholders = ",".join("?" * len(batch))
            rows = self.connection.execute(
                f"SELECT key, dim, vector FROM embeddings WHERE key IN ({placeholders})",
                batch,
            )
            for key, dim, blob in rows:
                found[key] = np.frombuffer(blob, dtype="float32", count=dim)
        return found

    def _store(self, keyed_vectors):
        self.connection.executemany(
            "INSERT OR REPLACE INTO embeddings (key, model, dim, vector) "
            "VALUES (?, ?, ?, ?)",
            [(key, self.model, int(vector.shape[0]),
              np.asarray(vector, dtype="float32").tobytes())
             for key, vector in keyed_vectors],
        )
        self.connection.commit()

    def embed(self, texts, compute, batch_size=4096):
        """Return one row per text, computing only the ones not already stored.

        `compute` takes a list of texts and returns an array of vectors.
        """
        if not texts:
            return np.empty((0, 0), dtype="float32")

        keys = [self.key_for(self.model, text) for text in texts]
        unique_keys = list(dict.fromkeys(keys))
        found = self._lookup(unique_keys)
        missing = [key for key in unique_keys if key not in found]

        if missing:
            text_by_key = {}
            for key, text in zip(keys, texts):
                text_by_key.setdefault(key, text)
            for start in range(0, len(missing), batch_size):
                batch = missing[start:start + batch_size]
                vectors = np.asarray(
                    compute([text_by_key[key] for key in batch]), dtype="float32"
                )
                if len(vectors) != len(batch):
                    raise RuntimeError(
                        f"embedding batch returned {len(vectors)} vectors "
                        f"for {len(batch)} texts"
                    )
                self._store(list(zip(batch, vectors)))
                found.update(dict(zip(batch, vectors)))

        self.hits += len(unique_keys) - len(missing)
        self.misses += len(missing)
        return np.vstack([found[key] for key in keys])

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
