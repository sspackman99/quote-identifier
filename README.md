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

## Corpora

Corpus text is external (see `data/README.md`). Each source has its own parser, and all of
them emit the same canonical record:

    corpus, book_id, book, chapter, verse, text, source_ref, source_line

- `corpus`: corpus id, e.g. `bom`, `kjv`
- `book_id`: stable identifier — `isaiah` from any source, whatever its heading style
- `chapter`, `verse`: integers
- `source_ref`, `source_line`: what the file actually printed and where, for audit

Registered parsers (`corpora.CORPUS_PARSERS`):

    bom   Project Gutenberg Book of Mormon (book titles, chapter headings, inline
          4 Nephi verse markers)
    kjv   one tab-separated verse per line: "Genesis 1:1<TAB>In the beginning..."

Parse one corpus through its own parser:

    python main.py --corpus data/bom.txt --corpus-id bom
    python main.py --corpus data/kjv.txt --corpus-id kjv

A line that is neither a header nor a verse raises `CorpusFormatError` naming the line
number, instead of being dropped — a truncated file must not look like a shorter canon.

### Cross-corpus retrieval

`cross_corpus.py` windows both corpora into one index and asks what a chapter of one
quotes from the other:

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python cross_corpus.py --bom data/bom.txt --kjv data/kjv.txt \
        --query "Mosiah 14" --top 10 --json cc_mosiah14.json

It reports, for each target chapter, how many candidate windows of the query chapter
pointed at it, with best and mean similarity, plus whether Isaiah 53 was retrieved.
Common trigrams are filtered once over the pooled corpus so both sides lose the same
boilerplate. MinHash is not used here: this stage is about embedding retrieval.

### Retrieval evaluation

`evaluate_retrieval.py` scores the retriever against a labelled set
(`evaluation_pairs.py`). Each pair carries the source and target **verse spans**
(`whole` or `21-24`), a confidence, a note when the label rests on a judgement, and a
`reviewed` flag showing whether the label has been checked against the text — 10 of 44
have been so far.

  quotation          21 pairs: whole chapters (Isaiah 48-49, 50-51, 2-14; Isaiah 53,
                     54; Malachi 3-4)
  partial_quotation   3 pairs: Mosiah 12:21-24, 3 Nephi 20:36-40, Moroni 10:31
  paraphrase          4 pairs: 3 Nephi 12-14 / Matthew 5-7, Moroni 7:45-48 / 1 Cor 13:4-8
  allusion            2 pairs: Alma 7:10-12, 3 Nephi 19:20-23
  interpretation      3 pairs: Mosiah 15, 2 Nephi 2:15-21, Alma 42:2-6 — contextual
                      dependence on a source the discourse already quotes or retells.
                      Scored and printed SEPARATELY, excluded from retrieval recall:
                      an exposition of Isaiah 53 does not "retrieve" it.
  shared_vocabulary   3 controls with KJV-shaped vocabulary and no source chapter
  unrelated           8 controls with no KJV source at all

The dev/test split is assigned **per source family**, not per query, so neighbouring
chapters of one quotation block cannot straddle it. Thresholds are swept on dev only;
test is reported at a threshold fixed in advance (0.90) with precision, recall and
false-positive rate, where an accepted control is a false positive and the count of
unreviewed labels is printed alongside.

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python evaluate_retrieval.py --cache-index .eval_cache --json eval_result.json

It builds one pooled index (cached in `.eval_cache/` after the first run, so re-scoring
is seconds rather than minutes), ranks KJV chapters for each query chapter by window
hits, and prints per-kind recall, the dev threshold sweep, and the held-out result.

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

## Filtering common phrases

3-grams that are too common are stripped before matching (`filtering.py`). Two measures
exist, and their thresholds are NOT interchangeable because the denominators differ:

- `document frequency` (default): how many chapters contain the trigram, counted once per
  chapter however often it repeats. A document is `(corpus, book, chapter)`, so the same
  book/chapter in a different corpus is a different document.
- `occurrence count` (legacy, `--filter-metric occurrence`): every position in every
  window. Overlapping windows and repeats inside a passage inflate it. Retained as a
  separate statistic so the two can be compared on the same corpus.

Measured on the Book of Mormon (239 chapters, 17,857 windows, 535,710 window tokens):

    metric                                    phrases   tokens removed
    document >20% of chapters (default)            76   52,059  (9.7%)
    occurrence >1% of windows (legacy)             54   45,927  (8.6%)
    document >1% of chapters (same number)     12,341  281,655 (52.6%)

That last row is why `MIN_DOCUMENT_FREQUENCY` is 0.20 and not 0.01: on this corpus 1% of
239 chapters means "in 3 chapters", which is a far looser bar than 1% of 17,857 windows.

Effect on the top 750: 606 of 665 reference pairs are shared; 59 appear only under the
document metric and 58 only under the legacy one, including some high-scoring pairs in
both directions. `--filter-metric occurrence` reproduces the old behaviour exactly.

## Tests

    python -m pytest -q

