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
import numpy as np

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
from similarity import DEFAULT_HTTP_MODEL, add_embeddings, build_faiss_index
from top_5_candidates import find_candidates_between, find_top_5_embedding_candidates

WINDOW_SIZE = 30
WINDOW_OVERLAP = 15
# Candidates kept per source window. Every retrieved candidate is written to the
# CSV — there is no global row cap — so this is the only thing that bounds the
# output size.
TOP_K = 5

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


def load_windows(corpus_id, path):
    """Parse and window one corpus; each window already carries its corpus id."""
    verses = preprocess_corpus(corpus_id, str(path), remove_punctuation=False)
    streams = build_chapter_token_stream(verses)
    windows = create_overlapping_windows(
        streams, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP
    )
    return verses, windows


def common_phrases(windows, corpus_ids):
    """Trigrams too common in ANY corpus, each judged against its own chapters.

    MIN_DOCUMENT_FREQUENCY is a fraction of a canon's chapters, so it is not
    portable across corpora: 20% of 239 Book of Mormon chapters is 48 chapters,
    20% of the 1,428 chapters in Book of Mormon + Bible is 285. One pooled count
    would let the commonest phrases through on one side, which is exactly what
    the filter exists to prevent in a cross-corpus comparison.
    """
    phrases = set()
    per_corpus = {}

    for corpus_id in corpus_ids:
        subset = [window for window in windows if window["corpus"] == corpus_id]
        frequencies, total = build_ngram_document_frequencies(subset, n=NGRAM_N)
        found = identify_ngrams_by_document_frequency(
            frequencies, total_documents=total,
            min_document_frequency=MIN_DOCUMENT_FREQUENCY,
        )
        per_corpus[corpus_id] = (len(found), total)
        phrases |= found

    return phrases, per_corpus


def embedding_model_name():
    """The model that embeddings actually come from, for cache keys and metadata.

    It has to match what similarity.embed_texts will use, or the cache would file
    vectors under a name nothing looks up.
    """
    configured = os.environ.get("QI_EMBEDDINGS_MODEL")
    if configured:
        return configured
    if os.environ.get("QI_EMBEDDINGS_URL"):
        return DEFAULT_HTTP_MODEL
    return LOCAL_EMBEDDING_MODEL


def build_state(corpora, filter_metric="document", cache=None):
    """Load, window, filter and embed every corpus once.

    `corpora` is [(corpus_id, path), ...]. The returned state carries the pooled
    window list, one embedding matrix, and the window positions belonging to each
    corpus, so any number of query/source pairs can be retrieved from it without
    preprocessing or embedding again.
    """
    windows = []
    corpora_meta = []

    for corpus_id, path in corpora:
        print(f"Preprocessing {corpus_id}...")
        verses, corpus_windows = load_windows(corpus_id, path)
        print(f"  verses: {len(verses)}, windows: {len(corpus_windows)}")
        corpora_meta.append({
            "corpus_id": corpus_id,
            "path": str(path),
            "verses": len(verses),
            "windows": len(corpus_windows),
        })
        windows.extend(corpus_windows)

    print("Computing n-gram frequencies...")
    occurrence_counts = build_ngram_frequencies(windows, n=NGRAM_N)
    document_frequency, total_documents = build_ngram_document_frequencies(
        windows, n=NGRAM_N
    )

    if filter_metric == "document":
        high_freq, per_corpus = common_phrases(
            windows, [meta["corpus_id"] for meta in corpora_meta]
        )
        described = ", ".join(
            f"{corpus_id} {count}/{total}"
            for corpus_id, (count, total) in per_corpus.items()
        )
        print(
            f"Filter metric: document frequency per corpus "
            f"(> {MIN_DOCUMENT_FREQUENCY:.0%} of that corpus's chapters: "
            f"{described}) -> {len(high_freq)} phrases"
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

    print("Computing embeddings...")
    windows = add_embeddings(windows, cache=cache)
    embeddings = np.array(
        [window["embedding"] for window in windows], dtype="float32"
    )

    positions = {}
    for position, window in enumerate(windows):
        positions.setdefault(window["corpus"], []).append(position)

    return {
        "windows": windows,
        "embeddings": embeddings,
        "positions": positions,
        "corpora": corpora_meta,
        "high_freq": high_freq,
        "total_documents": total_documents,
        "filter_metric": filter_metric,
        "filter_threshold": (MIN_DOCUMENT_FREQUENCY
                             if filter_metric == "document"
                             else LEGACY_OCCURRENCE_RATE),
    }


def pairs_for(state, query_id, source_id, top_k=TOP_K):
    """Ranked, collapsed pairs for one query/source pair of corpora.

    Across corpora the search covers ONLY the source corpus, so within-corpus
    near-duplicates cannot crowd the other canon out of the top-k. Within one
    corpus the query and target sets are the same windows, which is the original
    all-against-all behaviour.

    Returns (pairs, directed_candidate_count).
    """
    windows = state["windows"]
    positions = state["positions"]

    for corpus_id in (query_id, source_id):
        if corpus_id not in positions:
            raise ValueError(
                f"corpus {corpus_id!r} has no windows; saw {sorted(positions)}"
            )

    candidates = find_candidates_between(
        windows, state["embeddings"], positions[query_id], positions[source_id],
        top_k,
    )
    pairs = collapse_directions(candidates)
    print(f"  {query_id} -> {source_id}: {len(candidates)} candidate rows "
          f"-> {len(pairs)} pairs", flush=True)
    return pairs, len(candidates)


def write_pairs(state, pairs, output_path, query_id, source_id,
                top_k=TOP_K, pair_mode=None, directed_candidates=None):
    """Write one pair CSV plus the run metadata that produced it."""
    pairs = sorted(pairs, key=lambda entry: -entry["cosine_similarity"])
    df = build_candidate_rows(state["windows"], pairs)
    df.to_csv(output_path, index=False)
    print(f"CSV saved as {output_path} ({len(df)} rows)")

    corpora = [dict(meta) for meta in state["corpora"]]
    for entry in corpora:
        entry["digest"] = corpus_digest(entry["path"])

    query_meta = next(meta for meta in corpora if meta["corpus_id"] == query_id)

    write_run_metadata(
        output_path,
        corpus_id=query_id,
        corpus_path=query_meta["path"],
        corpus_digest=query_meta["digest"],
        pair={"query": query_id, "source": source_id},
        corpora=corpora,
        pair_mode=pair_mode or ("within" if query_id == source_id else "cross"),
        verses=sum(meta["verses"] for meta in corpora),
        windows=len(state["windows"]),
        filter_metric=state["filter_metric"],
        filter_threshold=state["filter_threshold"],
        filtered_phrases=len(state["high_freq"]),
        documents=state["total_documents"],
        ngram_n=NGRAM_N,
        window_size=WINDOW_SIZE,
        window_overlap=WINDOW_OVERLAP,
        scoring="cosine_similarity",
        top_k=top_k,
        directed_candidates=directed_candidates,
        rows=len(df),
        embedding_endpoint=os.environ.get("QI_EMBEDDINGS_URL", "local"),
        embedding_model=embedding_model_name(),
    )
    return df


def run(corpus_path, output_path="top_matches.csv",
        filter_metric="document", corpus_id="bom", top_k=TOP_K,
        corpus_b=None, corpus_id_b=None, pair_mode="both", cache=None):
    """Run the pipeline for one corpus, or two, and write a single CSV.

    This is the one-shot entry point. The matrix driver builds the state once and
    calls pairs_for/write_pairs per pair instead, which is what makes ten runs
    cost one embedding pass.
    """
    corpora = [(corpus_id, str(corpus_path))]
    if corpus_b:
        corpora.append((corpus_id_b or corpus_id, str(corpus_b)))
    if len(corpora) == 2 and corpora[0][0] == corpora[1][0]:
        raise ValueError(
            "a second corpus needs its own id (--corpus-id-b); two corpora "
            "sharing one id would merge into a single document space"
        )

    print("Corpora: " + ", ".join(cid for cid, _ in corpora))
    state = build_state(corpora, filter_metric=filter_metric, cache=cache)

    if len(corpora) == 1:
        jobs = [(corpora[0][0], corpora[0][0])]
    elif pair_mode == "within":
        jobs = [(cid, cid) for cid, _ in corpora]
    elif pair_mode == "cross":
        jobs = [(corpora[0][0], corpora[1][0])]
    else:  # both
        jobs = [(cid, cid) for cid, _ in corpora]
        jobs.append((corpora[0][0], corpora[1][0]))

    pairs, directed = [], 0
    for query_id, source_id in jobs:
        found, count = pairs_for(state, query_id, source_id, top_k)
        pairs.extend(found)
        directed += count

    print(f"Directed candidate rows: {directed}")
    print(f"Unique pairs: {len(pairs)}")

    return write_pairs(
        state, pairs, output_path,
        query_id=corpora[0][0],
        source_id=corpora[-1][0],
        top_k=top_k,
        pair_mode=pair_mode,
        directed_candidates=directed,
    )



CANDIDATE_COLUMNS = ["cosine_similarity", "rank_1", "rank_2"]
for _side in ("1", "2"):
    CANDIDATE_COLUMNS += [
        f"corpus_{_side}", f"book_id_{_side}", f"book_{_side}",
        f"chapter_{_side}", f"verse_{_side}", f"end_verse_{_side}",
        f"ref_{_side}", f"text_{_side}",
    ]


def collapse_directions(candidates):
    """One record per unordered window pair, keeping each side's rank.

    Cosine similarity is symmetric, so A->B and B->A carry the same score and
    the same two passages; only the rank differs, and it differs often. Both
    ranks are therefore kept as separate fields instead of emitting two rows.

    A pair is included whenever EITHER direction retrieved it, so collapsing
    cannot lose a pair that only one side saw. `ranks` is keyed by the window
    that did the ranking, and a side missing from it means that side ranked the
    pair outside its top-k.
    """
    merged = {}

    for candidate in candidates:
        a = candidate["source_window_id"]
        b = candidate["candidate_window_id"]
        key = (a, b) if a < b else (b, a)

        entry = merged.setdefault(key, {
            "pair": key,
            "cosine_similarity": candidate["cosine_similarity"],
            "ranks": {},
        })
        entry["ranks"][a] = candidate["candidate_rank"]

    return sorted(
        merged.values(),
        key=lambda entry: (-entry["cosine_similarity"], entry["pair"]),
    )


def select_pair_mode(pairs, windows, mode):
    """Keep same-corpus pairs ('within'), different-corpus pairs ('cross'), or all."""
    if mode == "both":
        return pairs

    want_cross = mode == "cross"
    kept = []
    for entry in pairs:
        a, b = entry["pair"]
        is_cross = windows[a]["corpus"] != windows[b]["corpus"]
        if is_cross == want_cross:
            kept.append(entry)
    return kept


def build_candidate_rows(windows, pairs):
    """Turn collapsed pair records into the output rows.

    One row per unordered pair, sides ordered by window index so the file is
    deterministic. `cosine_similarity` is the only score, and it is symmetric:
    the same value ranked the pair from either direction. `rank_1` is where
    side 2 stood in side 1's candidates, `rank_2` where side 1 stood in side
    2's; a blank rank means that side ranked the pair outside its top-k.
    """
    rows = []
    for entry in pairs:
        a, b = entry["pair"]
        row = {
            "cosine_similarity": entry["cosine_similarity"],
            "rank_1": entry["ranks"].get(a),
            "rank_2": entry["ranks"].get(b),
        }
        for side, window in (("1", windows[a]), ("2", windows[b])):
            row[f"corpus_{side}"] = window["corpus"]
            row[f"book_id_{side}"] = window.get("book_id")
            row[f"book_{side}"] = window["book"]
            row[f"chapter_{side}"] = window["chapter"]
            row[f"verse_{side}"] = window["start_verse"]
            row[f"end_verse_{side}"] = window["end_verse"]
            row[f"ref_{side}"] = format_reference(
                window["book"], window["chapter"],
                window["start_verse"], window["end_verse"],
            )
            row[f"text_{side}"] = window["text"]
        rows.append(row)

    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    # A rank the other side never assigned is a missing value, not zero. Int64
    # keeps it blank in the CSV instead of turning the column into floats and
    # writing "1.0" for every rank.
    for rank_column in ("rank_1", "rank_2"):
        df[rank_column] = df[rank_column].astype("Int64")
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
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"candidates written per source window (default: {TOP_K}). Every "
             "candidate is kept; there is no global row cap",
    )
    parser.add_argument(
        "--filter-metric",
        choices=["document", "occurrence"],
        default="document",
        help="how common a 3-gram must be before it is filtered out: 'document' "
             "(default) counts chapters containing it, 'occurrence' reproduces the "
             "legacy count-per-window rate",
    )
    parser.add_argument(
        "--corpus-b",
        default=None,
        help="second corpus to pool into the same index (cross-corpus retrieval)",
    )
    parser.add_argument(
        "--corpus-id-b",
        choices=available_corpora(),
        default=None,
        help="parser for --corpus-b (default: the same as --corpus-id)",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="SQLite embedding cache; repeat runs and matrix runs reuse vectors",
    )
    parser.add_argument(
        "--pairs",
        choices=["both", "within", "cross"],
        default="both",
        help="which pairs to write: same-corpus ('within'), different-corpus "
             "('cross'), or all (default). 'cross' needs --corpus-b",
    )
    args = parser.parse_args(argv)

    if args.pairs == "cross" and not args.corpus_b:
        parser.exit(2, f"{parser.prog}: error: --pairs cross needs --corpus-b\n")
    if args.corpus_b and args.corpus_b == args.corpus:
        parser.exit(2, f"{parser.prog}: error: --corpus-b must differ from --corpus\n")

    try:
        corpus_path = resolve_corpus_path(args.corpus)
    except FileNotFoundError as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    cache = None
    if args.cache:
        from embeddings_cache import EmbeddingCache
        cache = EmbeddingCache(args.cache, embedding_model_name())

    print(f"Corpus: {corpus_path} ({args.corpus_id})")
    run(corpus_path, args.output, filter_metric=args.filter_metric,
        corpus_id=args.corpus_id, top_k=args.top_k,
        corpus_b=args.corpus_b, corpus_id_b=args.corpus_id_b,
        pair_mode=args.pairs, cache=cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
