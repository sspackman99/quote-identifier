"""Evaluate cross-corpus retrieval against a labelled set of quotation pairs.

Builds ONE index over both corpora (or loads a cached one), then for every
labelled pair ranks KJV chapters by how many of the query chapter's windows
point at them. Reports recall@1/5/20 for quotations and paraphrases separately,
and shows what the retriever returns for the unlabelled controls.

Usage:
    python evaluate_retrieval.py --cache-index .eval_cache
    python evaluate_retrieval.py --json eval_result.json

The index cache makes re-running after a scoring change cheap: the first run
embeds both corpora (~6 min), later runs load the vectors in seconds.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from chunking import build_chapter_token_stream, create_overlapping_windows
from corpora import format_reference
from evaluation_pairs import all_pairs
from filtering import (
    build_ngram_document_frequencies,
    identify_ngrams_by_document_frequency,
    remove_high_freq_phrases,
)
from preprocess import preprocess_corpus
from similarity import add_embeddings

WINDOW_SIZE = 30
WINDOW_OVERLAP = 15
NGRAM_N = 3
MIN_DOCUMENT_FREQUENCY = 0.20
META_FIELDS = ("corpus", "book_id", "book", "chapter", "start_verse", "end_verse", "text")


def build_windows(bom_path, kjv_path):
    windows = []
    for corpus_id, path in (("bom", bom_path), ("kjv", kjv_path)):
        verses = preprocess_corpus(corpus_id, str(path))
        streams = build_chapter_token_stream(verses)
        windows.extend(
            create_overlapping_windows(streams, window_size=WINDOW_SIZE,
                                       overlap=WINDOW_OVERLAP)
        )
    return windows


def load_or_build_index(args):
    cache = Path(args.cache_index) if args.cache_index else None
    embeddings_path = cache / "embeddings.npy" if cache else None
    meta_path = cache / "windows.json" if cache else None

    if cache and embeddings_path.exists() and meta_path.exists():
        print(f"Loading cached index from {cache}")
        embeddings = np.load(embeddings_path)
        meta = json.loads(meta_path.read_text())
        return meta, embeddings

    print("Building index from scratch")
    windows = build_windows(args.bom, args.kjv)

    document_frequency, total_documents = build_ngram_document_frequencies(
        windows, n=NGRAM_N
    )
    common = identify_ngrams_by_document_frequency(
        document_frequency, total_documents, MIN_DOCUMENT_FREQUENCY
    )
    remove_high_freq_phrases(windows, common, n=NGRAM_N)
    print(f"Filtered {len(common)} common trigrams across {total_documents} chapters")

    print(f"Embedding {len(windows)} windows...")
    windows = add_embeddings(windows)
    embeddings = np.asarray([w["embedding"] for w in windows], dtype="float32")

    meta = [{field: w.get(field) for field in META_FIELDS} for w in windows]

    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings)
        meta_path.write_text(json.dumps(meta))
        print(f"Index cached in {cache}")

    return meta, embeddings


def rank_chapters(candidates):
    """Rank target chapters by hit count, then best similarity, then name."""
    ranked = [
        {"book": book, "chapter": chapter,
         "hits": len(scores), "best": max(scores), "mean": sum(scores) / len(scores)}
        for (book, chapter), scores in candidates.items()
    ]
    ranked.sort(key=lambda row: (-row["hits"], -row["best"], row["book"], row["chapter"]))
    return ranked


def collect_candidates(meta, index, embeddings, query_book, query_chapter, neighbors):
    """For every query window, collect the target-corpus chapters it points at."""
    query_indices = [
        i for i, window in enumerate(meta)
        if window["corpus"] == "bom"
        and window["book"] == query_book
        and window["chapter"] == query_chapter
    ]
    if not query_indices:
        raise SystemExit(f"no windows for {query_book} {query_chapter}")

    k = min(neighbors + 1, len(meta))
    candidates = defaultdict(list)

    for i in query_indices:
        D, I = index.search(np.array([embeddings[i]]), k)
        for score, j in zip(D[0], I[0]):
            j = int(j)
            if j == i:
                continue
            target = meta[j]
            if target["corpus"] == "bom":
                continue
            candidates[(target["book"], target["chapter"])].append(float(score))

    return candidates, len(query_indices)


def parse_ref(text):
    book, chapter = text.rsplit(" ", 1)
    if not chapter.isdigit():
        raise ValueError(f"bad reference {text!r}")
    return book, int(chapter)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", default="data/bom.txt")
    parser.add_argument("--kjv", default="data/kjv.txt")
    parser.add_argument("--cache-index", default=".eval_cache",
                        help="directory for cached embeddings/meta (default .eval_cache)")
    parser.add_argument("--neighbors", type=int, default=50,
                        help="neighbours per query window (default 50)")
    parser.add_argument("--json", help="write the full result as JSON here")
    args = parser.parse_args(argv)

    meta, embeddings = load_or_build_index(args)

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    pairs = all_pairs()
    results = []

    print(f"\nEvaluating {len(pairs)} labelled pairs against {len(meta)} windows\n")
    print(f"{'#':>3}  {'query':<14} {'expected':<13} {'kind':<11} {'rank':>5} "
          f"{'hits':>5} {'best':>7}  {'top-1':<18}")

    for number, pair in enumerate(pairs, 1):
        query_book, query_chapter = parse_ref(pair["query"])
        candidates, query_windows = collect_candidates(
            meta, index, embeddings, query_book, query_chapter, args.neighbors
        )
        ranked = rank_chapters(candidates)
        top = ranked[0] if ranked else None

        expected = pair["expected"]
        expected_rank = None
        expected_row = None
        if expected:
            expected_book, expected_chapter = parse_ref(expected)
            for rank, row in enumerate(ranked, 1):
                if row["book"] == expected_book and row["chapter"] == expected_chapter:
                    expected_rank, expected_row = rank, row
                    break

        results.append({
            "query": pair["query"],
            "expected": expected,
            "kind": pair["kind"],
            "query_windows": query_windows,
            "expected_rank": expected_rank,
            "expected_hits": expected_row["hits"] if expected_row else 0,
            "expected_best": expected_row["best"] if expected_row else None,
            "top1": f"{top['book']} {top['chapter']}" if top else None,
            "top1_best": top["best"] if top else None,
            "chapters_retrieved": len(ranked),
        })

        expected_hits = expected_row["hits"] if expected_row else 0
        expected_best = f"{expected_row['best']:.4f}" if expected_row else "-"
        rank_text = str(expected_rank) if expected_rank else "-"

        print(f"{number:>3}  {pair['query']:<14} {(expected or '-'):<13} {pair['kind']:<11} "
              f"{rank_text:>5} {expected_hits:>5} {expected_best:>7}  "
              f"{(results[-1]['top1'] or '-'):<18}")

    # aggregate
    summary = {}
    for kind in ("quotation", "paraphrase"):
        rows = [row for row in results if row["kind"] == kind and row["expected"]]
        if not rows:
            continue
        summary[kind] = {
            "pairs": len(rows),
            "recall_at_1": sum(1 for row in rows if row["expected_rank"] == 1) / len(rows),
            "recall_at_5": sum(1 for row in rows
                               if row["expected_rank"] and row["expected_rank"] <= 5) / len(rows),
            "recall_at_20": sum(1 for row in rows
                                if row["expected_rank"] and row["expected_rank"] <= 20) / len(rows),
            "missed": [row["query"] for row in rows if not row["expected_rank"]],
        }

    print("\nRecall by kind")
    for kind, values in summary.items():
        print(f"  {kind:<11} n={values['pairs']:<3} "
              f"@1={values['recall_at_1']:.0%}  @5={values['recall_at_5']:.0%}  "
              f"@20={values['recall_at_20']:.0%}"
              + (f"  misses={values['missed']}" if values["missed"] else ""))

    controls = [row for row in results if row["kind"] == "control"]
    if controls:
        print("\nControls (no KJV source; showing what was returned anyway)")
        for row in controls:
            best = row["top1_best"]
            print(f"  {row['query']:<12} top-1 {row['top1']:<16} "
                  f"best {best:.4f}  chapters hit {row['chapters_retrieved']}")

    if args.json:
        payload = {
            "windows": len(meta),
            "neighbors": args.neighbors,
            "window_size": WINDOW_SIZE,
            "window_overlap": WINDOW_OVERLAP,
            "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "results": results,
            "recall": summary,
            "controls": controls,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nJSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
