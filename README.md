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

## Output

`top_matches.csv` holds the highest-scoring candidate pairs:

    score, methods, text_1, book_1, chapter_1, start_verse_1, end_verse_1,
    text_2, book_2, chapter_2, start_verse_2, end_verse_2

`methods` lists every retriever that found the pair (`embedding`, `minhash`). Each pair
appears once: the two retrievers and both directions (A->B, B->A) are collapsed into one
row. `--no-dedupe` restores the raw per-method, per-direction rows.

Candidate pairs are still window pairs, and windows overlap, so the same passage pair can
legitimately appear more than once with different verse spans. Collapsing those belongs
with context expansion, where a span means something.

## Windowing

30-token windows with 15-token overlap, built per chapter (`chunking.py`):

- a window never crosses a chapter boundary;
- a chapter shorter than the window yields one window holding the whole chapter;
- the last window is anchored to the chapter end when the step grid would leave a tail,
  so no trailing text is left uncovered.

`test_window_coverage.py` checks those as geometric properties over many chapter lengths
and window/overlap settings, independently of the similarity pipeline.

## Tests

    python -m pytest -q

