"""Run the quote-identifier pipeline over a scripture corpus.

The corpus text file is not part of this repository (see data/README.md).
Resolve order for it: --corpus PATH, then $QI_CORPUS, then data/bom.txt,
then ./bom.txt.

Embeddings are computed locally when torch/sentence-transformers are available,
or fetched from an OpenAI-compatible /v1/embeddings endpoint named in
QI_EMBEDDINGS_URL (see README.md).
"""

import argparse
import sys

import pandas as pd

from chunking import build_chapter_token_stream, create_overlapping_windows
from filtering import (
    build_ngram_frequencies,
    identify_high_frequency_ngrams,
    remove_high_freq_phrases,
)
from preprocess import preprocess_bom, resolve_corpus_path
from similarity import (
    add_embeddings,
    add_minhashes,
    build_faiss_index,
    build_lsh,
    filter_nearby_matches,
    find_embedding_matches,
    find_minhash_matches,
    get_top_matches,
    merge_duplicate_matches,
)

WINDOW_SIZE = 30
WINDOW_OVERLAP = 15
TOP_N = 750


def run(corpus_path, output_path="top_matches.csv", dedupe=True):
    """Run the full pipeline and write the scored matches to a CSV."""
    # --------------------------------------------
    # 1. PREPROCESS
    # --------------------------------------------
    print("Preprocessing...")
    corpus_data = preprocess_bom(str(corpus_path), remove_punctuation=False)

    # --------------------------------------------
    # 2. CHUNKING
    # --------------------------------------------
    print("Creating overlapping windows...")
    chapter_streams = build_chapter_token_stream(corpus_data)
    windows = create_overlapping_windows(
        chapter_streams,
        window_size=WINDOW_SIZE,
        overlap=WINDOW_OVERLAP,
    )

    # --------------------------------------------
    # 3. REMOVE HIGH-FREQUENCY PHRASES
    # --------------------------------------------
    print("Computing n-gram frequencies...")
    ngram_counts = build_ngram_frequencies(windows, n=3)

    high_freq = identify_high_frequency_ngrams(
        ngram_counts,
        total_windows=len(windows),
        frequency_threshold=0.01,
    )

    print(f"High-frequency phrases identified: {len(high_freq)}")

    windows = remove_high_freq_phrases(windows, high_freq, n=3)

    # --------------------------------------------
    # 4. MINHASH MATCHING
    # --------------------------------------------
    print("Running MinHash matching...")
    windows = add_minhashes(windows)

    lsh = build_lsh(windows, threshold=0.6)
    minhash_matches = find_minhash_matches(windows, lsh)

    print(f"Raw MinHash matches: {len(minhash_matches)}")

    # --------------------------------------------
    # 5. EMBEDDING MATCHING
    # --------------------------------------------
    print("Computing embeddings...")
    windows = add_embeddings(windows)

    index, embeddings = build_faiss_index(windows)

    embedding_matches = find_embedding_matches(windows, index, embeddings, k=5)

    print(f"Raw embedding matches: {len(embedding_matches)}")

    # --------------------------------------------
    # 6. MERGE + FILTER
    # --------------------------------------------
    all_matches = minhash_matches + embedding_matches

    if dedupe:
        all_matches = merge_duplicate_matches(all_matches)
        print(f"Unique candidate pairs: {len(all_matches)}")

    print("Filtering nearby trivial matches...")
    filtered_matches = filter_nearby_matches(
        windows,
        all_matches,
        min_token_distance=200,
    )

    print(f"Filtered matches: {len(filtered_matches)}")

    # --------------------------------------------
    # 7. SCORE + OUTPUT CSV
    # --------------------------------------------
    print("Scoring matches and writing CSV...")

    top_matches = get_top_matches(windows, filtered_matches, top_n=TOP_N)

    rows = []
    for score, i, j, method in top_matches:
        w1 = windows[i]
        w2 = windows[j]
        methods = method if isinstance(method, str) else ",".join(sorted(method))

        rows.append({
            "score": score,
            "methods": methods,
            "text_1": w1["text"],
            "book_1": w1["book"],
            "chapter_1": w1["chapter"],
            "start_verse_1": w1["start_verse"],
            "end_verse_1": w1["end_verse"],
            "text_2": w2["text"],
            "book_2": w2["book"],
            "chapter_2": w2["chapter"],
            "start_verse_2": w2["start_verse"],
            "end_verse_2": w2["end_verse"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"CSV saved as {output_path}")
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find passages in a scripture corpus that look like quotations of each other."
    )
    parser.add_argument(
        "--corpus",
        help="path to the corpus text file (default: $QI_CORPUS, data/bom.txt, then ./bom.txt)",
    )
    parser.add_argument(
        "--output",
        default="top_matches.csv",
        help="CSV file to write (default: top_matches.csv)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="keep duplicate pair records (both directions, one per method) instead of "
             "collapsing each pair to one row",
    )
    args = parser.parse_args(argv)

    try:
        corpus_path = resolve_corpus_path(args.corpus)
    except FileNotFoundError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(f"Corpus: {corpus_path}")
    run(corpus_path, args.output, dedupe=not args.no_dedupe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
