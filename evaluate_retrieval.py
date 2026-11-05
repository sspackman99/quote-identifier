"""Evaluate cross-corpus retrieval against labelled textual relationships.

Builds ONE index over both corpora (or loads a cached one), then for every pair
ranks KJV chapters by how many of the query chapter's windows point at them.

Reporting rules that matter for honesty:

  * recall is computed over RETRIEVAL_KINDS only. "interpretation" pairs (an
    exposition of a source the discourse already quotes) are scored and shown
    separately, because they test contextual dependence, not retrieval.
  * thresholds are swept on the DEV split only; the held-out TEST split is
    reported at a threshold fixed in advance (0.90).
  * precision, recall and false-positive rate appear together, controls accepted
    above the threshold count as false positives, and the number of labels Sam has
    actually reviewed is printed next to the result.

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
from evaluation_pairs import (
    CONTEXT_KINDS,
    CONTROL_KINDS,
    RETRIEVAL_KINDS,
    all_pairs,
)
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

# Fixed before the held-out split was scored; not tuned on test data.
HELD_OUT_THRESHOLD = 0.90


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


def recall_metrics(rows):
    """Recall@1/5/20 over RETRIEVAL positives (interpretation is excluded)."""
    scored = [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
    if not scored:
        return None

    def at(k):
        return sum(1 for row in scored
                   if row["expected_rank"] and row["expected_rank"] <= k) / len(scored)

    return {
        "pairs": len(scored),
        "recall_at_1": at(1),
        "recall_at_5": at(5),
        "recall_at_20": at(20),
        "missed": [row["query"] for row in scored if not row["expected_rank"]],
    }


def threshold_metrics(rows, threshold):
    """Auto-accept every pair whose top hit scores at or above the threshold.

    tp              retrieval positive whose top hit is the labelled source
    fp_controls     control accepted — this is what the false-positive rate counts
    fp_context      interpretation accepted: a real relationship, but not a
                    quotation, so a wrong auto-accept that is not mere noise
    """
    scored = [row for row in rows if row["top1_best"] is not None]
    if not scored:
        return None

    accepted = [row for row in scored if row["top1_best"] >= threshold]
    true_positives = [row for row in accepted
                      if row["kind"] in RETRIEVAL_KINDS and row["top1"] == row["expected"]]
    fp_controls = [row for row in accepted if row["kind"] in CONTROL_KINDS]
    fp_context = [row for row in accepted if row["kind"] in CONTEXT_KINDS]
    other = [row for row in accepted
             if row not in true_positives
             and row not in fp_controls
             and row not in fp_context]

    retrieval_positives = [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
    controls = [row for row in rows if row["kind"] in CONTROL_KINDS]

    return {
        "threshold": round(threshold, 3),
        "accepted": len(accepted),
        "true_positives": len(true_positives),
        "false_positives_controls": len(fp_controls),
        "accepted_context": len(fp_context),
        "false_positives_other": len(other),
        "precision": (len(true_positives) / len(accepted)) if accepted else None,
        "recall": (len(true_positives) / len(retrieval_positives))
                  if retrieval_positives else None,
        "false_positive_rate": (len(fp_controls) / len(controls)) if controls else None,
    }


def threshold_sweep(rows, step=0.02, low=0.60, high=1.00):
    sweep = []
    threshold = low
    while threshold <= high + 1e-9:
        metrics = threshold_metrics(rows, threshold)
        if metrics:
            sweep.append(metrics)
        threshold = round(threshold + step, 4)
    return sweep


def print_recall(title, rows):
    print(f"\n{title}")
    for kind in RETRIEVAL_KINDS:
        metrics = recall_metrics([row for row in rows if row["kind"] == kind])
        if not metrics:
            continue
        print(f"  {kind:<18} n={metrics['pairs']:<3} @1={metrics['recall_at_1']:.0%}  "
              f"@5={metrics['recall_at_5']:.0%}  @20={metrics['recall_at_20']:.0%}"
              + (f"  misses={metrics['missed']}" if metrics["missed"] else ""))
    overall = recall_metrics(rows)
    if overall:
        print(f"  {'retrieval total':<18} n={overall['pairs']:<3} "
              f"@1={overall['recall_at_1']:.0%}  @5={overall['recall_at_5']:.0%}  "
              f"@20={overall['recall_at_20']:.0%}")


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
    reviewed = sum(1 for pair in pairs if pair["reviewed"])

    print(f"\nEvaluating {len(pairs)} labelled pairs against {len(meta)} windows "
          f"({reviewed} reviewed against the text, {len(pairs) - reviewed} not)")
    print(f"{'#':>3}  {'query':<12} {'span':<10} {'expected':<15} {'span':<11} "
          f"{'kind':<18} {'split':<5} {'rank':>4} {'best':>7}  {'top-1':<18}")

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
            "family": pair["family"],
            "split": pair["split"],
            "reviewed": pair["reviewed"],
            "confidence": pair["confidence"],
            "query_span": pair["query_span"],
            "expected_span": pair["expected_span"],
            "note": pair["note"],
            "query_windows": query_windows,
            "expected_rank": expected_rank,
            "expected_hits": expected_row["hits"] if expected_row else 0,
            "expected_best": expected_row["best"] if expected_row else None,
            "top1": f"{top['book']} {top['chapter']}" if top else None,
            "top1_best": top["best"] if top else None,
            "chapters_retrieved": len(ranked),
        })

        rank_text = str(expected_rank) if expected_rank else "-"
        best_text = f"{expected_row['best']:.4f}" if expected_row else "-"
        print(f"{number:>3}  {pair['query']:<12} {pair['query_span']:<10} "
              f"{(expected or '-'):<15} {pair['expected_span']:<11} "
              f"{pair['kind']:<18} {pair['split']:<5} {rank_text:>4} {best_text:>7}  "
              f"{(results[-1]['top1'] or '-'):<18}")

    dev = [row for row in results if row["split"] == "dev"]
    test = [row for row in results if row["split"] == "test"]

    print_recall("Recall — dev split (tuning data)", dev)
    print_recall("Recall — test split (held out)", test)

    context_rows = [row for row in results if row["kind"] in CONTEXT_KINDS]
    if context_rows:
        print("\nContextual dependence, scored separately (not in recall)")
        for row in context_rows:
            if row["expected_best"] is None:
                print(f"  {row['query']:<12} -> {row['expected']:<12} not retrieved")
            else:
                print(f"  {row['query']:<12} -> {row['expected']:<12} rank "
                      f"{row['expected_rank'] or '-'} best {row['expected_best']:.4f}")

    sweep_rows = threshold_sweep(dev)
    if sweep_rows:
        print(f"\nThreshold sweep on DEV only ({len(sweep_rows)} points)")
        print(f"  {'thresh':>6} {'accept':>6} {'tp':>4} {'fpC':>4} {'ctx':>4} "
              f"{'other':>5} {'prec':>5} {'rec':>5} {'fpr':>5}")
        for row in sweep_rows:
            if row["accepted"] == 0:
                continue
            print(f"  {row['threshold']:>6.2f} {row['accepted']:>6} "
                  f"{row['true_positives']:>4} {row['false_positives_controls']:>4} "
                  f"{row['accepted_context']:>4} {row['false_positives_other']:>5} "
                  f"{row['precision']:>5.2f} {row['recall']:>5.2f} "
                  f"{row['false_positive_rate']:>5.2f}")

    held_out = threshold_metrics(test, HELD_OUT_THRESHOLD)
    if held_out:
        print(f"\nHeld-out TEST at the threshold fixed in advance "
              f"({HELD_OUT_THRESHOLD:.2f}):")
        print(f"  accepted {held_out['accepted']} of {len(test)} | tp "
              f"{held_out['true_positives']} | fp-controls "
              f"{held_out['false_positives_controls']} | accepted-context "
              f"{held_out['accepted_context']} | precision {held_out['precision']:.2f} "
              f"| recall {held_out['recall']:.2f} | fpr "
              f"{held_out['false_positive_rate']:.2f}")

    control_rows = [row for row in results if row["kind"] in CONTROL_KINDS]
    if control_rows:
        print("\nControls (what came back anyway)")
        for row in control_rows:
            flag = "" if row["reviewed"] else "  [unreviewed]"
            print(f"  {row['query']:<12} [{row['kind']:<17}] top-1 "
                  f"{row['top1']:<17} best {row['top1_best']:.4f}  "
                  f"chapters hit {row['chapters_retrieved']}{flag}")

    if args.json:
        payload = {
            "windows": len(meta),
            "neighbors": args.neighbors,
            "window_size": WINDOW_SIZE,
            "window_overlap": WINDOW_OVERLAP,
            "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "held_out_threshold": HELD_OUT_THRESHOLD,
            "labels_reviewed": reviewed,
            "labels_total": len(pairs),
            "results": results,
            "recall": {
                "dev": recall_metrics(dev),
                "test": recall_metrics(test),
                "by_kind": {kind: recall_metrics([row for row in results
                                                  if row["kind"] == kind])
                            for kind in RETRIEVAL_KINDS},
            },
            "contextual": context_rows,
            "threshold_sweep_dev": sweep_rows,
            "held_out_test": held_out,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nJSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
