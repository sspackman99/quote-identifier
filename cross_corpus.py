"""Cross-corpus retrieval: find the passages in one corpus that another quotes.

Both corpora are parsed into canonical records, windowed, embedded into ONE
index, and then a chosen chapter in the query corpus is used to retrieve its
nearest neighbours in the other corpus. This is the first end-to-end check that
the project can answer "what is this passage quoting?" rather than "what else in
this book looks similar?".

Usage:
    python cross_corpus.py --bom data/bom.txt --kjv data/kjv.txt \
        --query "Mosiah 14" --top 10

Embeddings come from QI_EMBEDDINGS_URL when set (OpenAI-compatible server),
otherwise from sentence-transformers locally.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from chunking import build_chapter_token_stream, create_overlapping_windows
from corpora import format_reference
from filtering import (
    build_ngram_document_frequencies,
    identify_ngrams_by_document_frequency,
    remove_high_freq_phrases,
)
from preprocess import preprocess_corpus
from similarity import add_embeddings, build_faiss_index

WINDOW_SIZE = 30
WINDOW_OVERLAP = 15
NGRAM_N = 3
MIN_DOCUMENT_FREQUENCY = 0.20


def windows_for(corpus_id, path):
    verses = preprocess_corpus(corpus_id, str(path))
    streams = build_chapter_token_stream(verses)
    windows = create_overlapping_windows(streams, window_size=WINDOW_SIZE,
                                         overlap=WINDOW_OVERLAP)
    return verses, windows


def strip_common_phrases(windows):
    """One shared filter over the pooled corpus, so both sides lose the same boilerplate."""
    document_frequency, total_documents = build_ngram_document_frequencies(
        windows, n=NGRAM_N
    )
    common = identify_ngrams_by_document_frequency(
        document_frequency, total_documents, MIN_DOCUMENT_FREQUENCY
    )
    remove_high_freq_phrases(windows, common, n=NGRAM_N)
    return len(common), total_documents


def parse_query(query):
    """'Mosiah 14' -> ('Mosiah', 14)."""
    parts = query.rsplit(" ", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise SystemExit(f"query must look like 'Mosiah 14', got {query!r}")
    return parts[0], int(parts[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", default="data/bom.txt", help="Book of Mormon text")
    parser.add_argument("--kjv", default="data/kjv.txt", help="King James text")
    parser.add_argument("--query", default="Mosiah 14",
                        help="chapter to look up, e.g. 'Mosiah 14' (default)")
    parser.add_argument("--top", type=int, default=10,
                        help="how many target chapters to report (default 10)")
    parser.add_argument("--neighbors", type=int, default=25,
                        help="neighbours to fetch per query window (default 25)")
    parser.add_argument("--json", help="also write the result as JSON here")
    args = parser.parse_args(argv)

    query_book, query_chapter = parse_query(args.query)

    print(f"Parsing Book of Mormon: {args.bom}")
    bom_verses, bom_windows = windows_for("bom", args.bom)
    print(f"Parsing King James Bible: {args.kjv}")
    kjv_verses, kjv_windows = windows_for("kjv", args.kjv)

    windows = bom_windows + kjv_windows
    print(f"Verses: {len(bom_verses)} bom + {len(kjv_verses)} kjv")
    print(f"Windows: {len(bom_windows)} bom + {len(kjv_windows)} kjv = {len(windows)}")

    phrases, documents = strip_common_phrases(windows)
    print(f"Filtered {phrases} common trigrams across {documents} chapters")

    print("Computing embeddings...")
    windows = add_embeddings(windows)
    index, embeddings = build_faiss_index(windows)

    # Tokens are only needed for filtering and embedding; drop them so the
    # 70k-window bookkeeping stays small.
    for window in windows:
        window.pop("tokens", None)
        window.pop("filtered_tokens", None)
        window.pop("filtered_text", None)

    # Query windows: the requested chapter of the Book of Mormon.
    query_indices = [
        i for i, window in enumerate(windows)
        if window["corpus"] == "bom"
        and window["book"] == query_book
        and window["chapter"] == query_chapter
    ]
    if not query_indices:
        raise SystemExit(f"no windows found for {query_book} {query_chapter}")

    print(f"\nQuery: {query_book} {query_chapter} "
          f"({len(query_indices)} windows, {len(embeddings[0])}-dim)")

    k = min(args.neighbors + 1, len(windows))
    chapter_scores = defaultdict(list)
    window_hits = 0
    covered_verses = set()

    for i in query_indices:
        D, I = index.search(np.array([embeddings[i]]), k)
        found_other_corpus = False
        for score, j in zip(D[0], I[0]):
            j = int(j)
            if j == i:
                continue
            target = windows[j]
            if target["corpus"] == "bom":
                continue  # cross-corpus only: this test is about the other corpus
            chapter_scores[(target["book"], target["chapter"])].append(float(score))
            found_other_corpus = True
            covered_verses.add((query_book, query_chapter, target["book"], target["chapter"]))
        if found_other_corpus:
            window_hits += 1

    if not chapter_scores:
        print("No cross-corpus neighbours found. Check the embeddings server / model.")
        return 1

    ranked = sorted(
        (
            (book, chapter, len(scores), max(scores), sum(scores) / len(scores))
            for (book, chapter), scores in chapter_scores.items()
        ),
        key=lambda row: (-row[2], -row[3], row[0], row[1]),
    )

    print(f"Query windows with at least one cross-corpus neighbour: "
          f"{window_hits}/{len(query_indices)}\n")
    print(f"{'rank':>4}  {'chapter':<22} {'hits':>5} {'best':>7} {'mean':>7}")
    for rank, (book, chapter, hits, best, mean) in enumerate(ranked[:args.top], 1):
        print(f"{rank:>4}  {book + ' ' + str(chapter):<22} {hits:>5} "
              f"{best:>7.4f} {mean:>7.4f}")

    isaiah_rank = next(
        (rank for rank, row in enumerate(ranked, 1) if row[0] == "Isaiah" and row[1] == 53),
        None,
    )
    print()
    if isaiah_rank:
        hits = next(row[2] for row in ranked if row[0] == "Isaiah" and row[1] == 53)
        print(f"Isaiah 53 retrieved: chapter rank {isaiah_rank} "
              f"({hits} query-window hits)")
    else:
        print("Isaiah 53 NOT among the retrieved chapters")

    if args.json:
        payload = {
            "query": f"{query_book} {query_chapter}",
            "query_windows": len(query_indices),
            "windows_total": len(windows),
            "filtered_trigrams": phrases,
            "documents": documents,
            "window_size": WINDOW_SIZE,
            "window_overlap": WINDOW_OVERLAP,
            "neighbors": args.neighbors,
            "query_windows_with_hits": window_hits,
            "isaiah_53_rank": isaiah_rank,
            "isaiah_53_hits": next(
                (row[2] for row in ranked if row[0] == "Isaiah" and row[1] == 53), 0
            ),
            "top": [
                {"rank": rank, "chapter": f"{book} {chapter}", "hits": hits,
                 "best": round(best, 6), "mean": round(mean, 6)}
                for rank, (book, chapter, hits, best, mean) in enumerate(ranked[:args.top], 1)
            ],
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"JSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
