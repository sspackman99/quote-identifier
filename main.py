"""Run the quote-identifier pipeline over a scripture corpus.

The corpus text file is not part of this repository (see data/README.md).
Resolve order for it: --corpus PATH, then $QI_CORPUS, then data/bom.txt,
then ./bom.txt.

Embeddings are computed locally when torch/sentence-transformers are available,
or fetched from an OpenAI-compatible /v1/embeddings endpoint named in
QI_EMBEDDINGS_URL (see README.md).
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from chunking import build_chapter_token_stream, create_overlapping_windows
from corpora import available_corpora, corpus_digest, format_reference
from filtering import (
    build_ngram_document_frequencies,
    build_ngram_frequencies,
    identify_high_frequency_ngrams,
    identify_ngrams_by_document_frequency,
    remove_high_freq_phrases,
)
from preprocess import preprocess_corpus, resolve_corpus_path
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

NGRAM_N = 3
# Fraction of DOCUMENTS (chapters) that must contain an n-gram before it is
# filtered out. Used when filter_metric == "document".
#
# Calibrated on the Book of Mormon (239 chapters, 17,857 windows): 0.20 selects
# 76 trigrams and drops 9.7% of window tokens, close to the legacy occurrence
# rule's 54 trigrams / 8.6%. The same 0.01 used on the occurrence scale would
# here select 12,341 trigrams and drop 52% of the text — the two denominators
# are not interchangeable, so the threshold has to be chosen per metric.
MIN_DOCUMENT_FREQUENCY = 0.20
# Fraction of WINDOWS, legacy occurrence-rate semantics. Used only to reproduce
# the pre-document-frequency behaviour.
LEGACY_OCCURRENCE_RATE = 0.01
# Recorded in run metadata when embeddings are computed locally.
LOCAL_EMBEDDING_MODEL = "all-mpnet-base-v2"


def run(corpus_path, output_path="top_matches.csv", dedupe=True,
        filter_metric="document", corpus_id="bom"):
    """Run the full pipeline and write the scored matches to a CSV."""
    # --------------------------------------------
    # 1. PREPROCESS
    # --------------------------------------------
    print("Preprocessing...")
    corpus_data = preprocess_corpus(
        corpus_id, str(corpus_path), remove_punctuation=False
    )
    print(f"Verses: {len(corpus_data)} ({corpus_id})")

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
    occurrence_counts = build_ngram_frequencies(windows, n=NGRAM_N)
    document_frequency, total_documents = build_ngram_document_frequencies(
        windows, n=NGRAM_N
    )

    if filter_metric == "document":
        high_freq = identify_ngrams_by_document_frequency(
            document_frequency,
            total_documents=total_documents,
            min_document_frequency=MIN_DOCUMENT_FREQUENCY,
        )
        print(
            f"Filter metric: document frequency "
            f"(> {MIN_DOCUMENT_FREQUENCY:.1%} of {total_documents} chapters) "
            f"-> {len(high_freq)} phrases"
        )
    elif filter_metric == "occurrence":
        high_freq = identify_high_frequency_ngrams(
            occurrence_counts,
            total_windows=len(windows),
            frequency_threshold=LEGACY_OCCURRENCE_RATE,
        )
        print(
            f"Filter metric: occurrence rate, legacy "
            f"(> {LEGACY_OCCURRENCE_RATE:.1%} of {len(windows)} windows) "
            f"-> {len(high_freq)} phrases"
        )
    else:
        raise ValueError(f"unknown filter metric: {filter_metric}")

    tokens_before = sum(len(window["tokens"]) for window in windows)
    windows = remove_high_freq_phrases(windows, high_freq, n=NGRAM_N)
    tokens_after = sum(len(window["filtered_tokens"]) for window in windows)
    print(f"Filtered tokens: {tokens_before - tokens_after} of {tokens_before} "
          f"({(tokens_before - tokens_after) / tokens_before:.1%})")

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

        row = {"score": score, "methods": methods}
        for prefix, window in (("1", w1), ("2", w2)):
            row[f"corpus_{prefix}"] = window["corpus"]
            row[f"book_id_{prefix}"] = window.get("book_id")
            row[f"book_{prefix}"] = window["book"]
            row[f"chapter_{prefix}"] = window["chapter"]
            row[f"verse_{prefix}"] = window["start_verse"]
            row[f"end_verse_{prefix}"] = window["end_verse"]
            row[f"ref_{prefix}"] = format_reference(
                window["book"], window["chapter"],
                window["start_verse"], window["end_verse"],
            )
            row[f"text_{prefix}"] = window["text"]
        rows.append(row)

    df = pd.DataFrame(rows)

    column_order = ["score", "methods"]
    for prefix in ("1", "2"):
        column_order += [f"corpus_{prefix}", f"book_id_{prefix}", f"book_{prefix}",
                         f"chapter_{prefix}", f"verse_{prefix}", f"end_verse_{prefix}",
                         f"ref_{prefix}", f"text_{prefix}"]
    df = df.reindex(columns=column_order)

    df.to_csv(output_path, index=False)
    print(f"CSV saved as {output_path}")

    write_run_metadata(
        output_path,
        corpus_id=corpus_id,
        corpus_path=str(corpus_path),
        corpus_digest=corpus_digest(corpus_path),
        verses=len(corpus_data),
        windows=len(windows),
        filter_metric=filter_metric,
        filter_threshold=(MIN_DOCUMENT_FREQUENCY if filter_metric == "document"
                          else LEGACY_OCCURRENCE_RATE),
        filtered_phrases=len(high_freq),
        documents=total_documents,
        ngram_n=NGRAM_N,
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        dedupe=dedupe,
        top_n=TOP_N,
        rows=len(df),
        embedding_endpoint=os.environ.get("QI_EMBEDDINGS_URL", "local"),
        embedding_model=os.environ.get("QI_EMBEDDINGS_MODEL", LOCAL_EMBEDDING_MODEL),
    )
    return df


def write_run_metadata(output_path, **facts) -> Path:
    """Write <output>.meta.json so a CSV can always be traced to its settings."""
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        **facts,
    }
    meta_path = Path(str(output_path) + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Run metadata saved as {meta_path}")
    return meta_path


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"



def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Find passages in a scripture corpus that look like quotations of each other."
    )
    parser.add_argument(
        "--corpus",
        help="path to the corpus text file (default: $QI_CORPUS, data/bom.txt, then ./bom.txt)",
    )
    parser.add_argument(
        "--corpus-id",
        choices=available_corpora(),
        default="bom",
        help="which parser to use for the file (default: bom)",
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
    parser.add_argument(
        "--filter-metric",
        choices=["document", "occurrence"],
        default="document",
        help="how common a 3-gram must be before it is filtered out: 'document' "
             "(default) counts chapters containing it, 'occurrence' reproduces the "
             "legacy count-per-window rate",
    )
    args = parser.parse_args(argv)

    try:
        corpus_path = resolve_corpus_path(args.corpus)
    except FileNotFoundError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(f"Corpus: {corpus_path} ({args.corpus_id})")
    run(corpus_path, args.output, dedupe=not args.no_dedupe,
        filter_metric=args.filter_metric, corpus_id=args.corpus_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
