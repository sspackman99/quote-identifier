# quote-identifier
A repository to identify moments in the scriptures where a speaker/author quotes another

## Running

The scripture text is not in this repository. Put your copy at `data/bom.txt`, pass
`--corpus PATH`, or set `QI_CORPUS`; see `data/README.md`. With no argument the pipeline
looks for `data/bom.txt`, then `./bom.txt`, and exits with an error naming both.

On a machine with a GPU (embeddings computed locally with sentence-transformers):

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    python main.py --corpus data/bom.txt --output top_matches.csv

On a CPU-only machine, with embeddings served by a GPU host over an
OpenAI-compatible `POST /v1/embeddings` endpoint:

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python main.py --corpus data/bom.txt

With `QI_EMBEDDINGS_URL` set (or `add_embeddings(..., url=...)` passed), torch and
sentence-transformers are never imported, so `numpy pandas faiss-cpu datasketch` is the
whole local requirement. Vectors are L2-normalized client-side, so cosine similarity
stays a plain inner product.

One index must be built from ONE embedding model. Mixing vectors from different models
(e.g. `all-mpnet-base-v2` locally and `embeddinggemma-300m` from a server) inside a
single run is meaningless, so pick one per run.

## Tests

    python -m pytest -q

