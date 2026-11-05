"""Evaluate cross-corpus retrieval against labelled textual relationships.

Builds ONE index over both corpora (or loads a cached one). For every pair, the
SAME matched-window evidence is ranked by two rules so the comparison is
like-for-like:

  chapter   the original rule: window-hit count, then best similarity
  span      span-aware: best similarity, then distinct non-overlapping supports,
            then coverage (see candidates.py)

The report shows both ranks per pair, whether the matched source range touches the
annotated span, and whether the query-side range touches the verses annotated as
doing the quoting. Annotations only SCORE the retrieval; nothing is told where the
quotation is.

Reporting rules: recall covers RETRIEVAL_KINDS only ("interpretation" is scored
separately); thresholds are swept on dev and the held-out split is reported at a
threshold fixed in advance (0.90); precision/recall/false-positive rate appear
together with the number of reviewed labels.

Usage:
    python evaluate_retrieval.py --cache-index .eval_cache
    python evaluate_retrieval.py --json eval_result.json --ranking span
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from candidates import (
    aggregate_candidates,
    collect_matches,
    rank_candidates,
    query_span_hit,
    span_hit,
    verses_per_chapter,
)
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
REGRESSION_CASES = ("Moroni 10", "Mosiah 12", "3 Nephi 20", "Mosiah 14")


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


def parse_ref(text):
    book, chapter = text.rsplit(" ", 1)
    if not chapter.isdigit():
        raise ValueError(f"bad reference {text!r}")
    return book, int(chapter)


def chapter_hits(meta, matches):
    """The original rule's input: chapter -> list of similarities."""
    hits = defaultdict(list)
    for match in matches:
        target = meta[match["target_index"]]
        hits[(target["book"], target["chapter"])].append(match["sim"])
    return hits


def rank_chapters(hits):
    """Original rule: most window hits first, then best similarity."""
    ranked = [
        {"book": book, "chapter": chapter,
         "hits": len(scores), "best": max(scores), "mean": sum(scores) / len(scores)}
        for (book, chapter), scores in hits.items()
    ]
    ranked.sort(key=lambda row: (-row["hits"], -row["best"], row["book"], row["chapter"]))
    return ranked


def recall_metrics(rows, rank_field="span_rank"):
    """Recall@1/5/20 over RETRIEVAL positives (interpretation is excluded)."""
    scored = [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
    if not scored:
        return None

    def at(k):
        return sum(1 for row in scored
                   if row.get(rank_field) and row[rank_field] <= k) / len(scored)

    return {
        "pairs": len(scored),
        "recall_at_1": at(1),
        "recall_at_5": at(5),
        "recall_at_20": at(20),
        "missed": [row["query"] for row in scored if not row.get(rank_field)],
    }


def span_recall_metrics(rows, rank_field="span_rank"):
    """Recall@k requiring the correct chapter in top k AND a source-span hit."""
    scored = [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
    if not scored:
        return None

    def at(k):
        return sum(1 for row in scored
                   if row.get(rank_field) and row[rank_field] <= k
                   and row.get("source_span_hit")) / len(scored)

    return {"pairs": len(scored), "recall_at_1": at(1),
            "recall_at_5": at(5), "recall_at_20": at(20)}


def localization_metrics(rows, rank_field="span_rank"):
    """Recall@k where the QUERY-side supporting range touches the annotated verses."""
    scored = [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
    if not scored:
        return None

    def at(k):
        return sum(1 for row in scored
                   if row.get(rank_field) and row[rank_field] <= k
                   and row.get("query_span_hit")) / len(scored)

    return {"pairs": len(scored), "recall_at_1": at(1),
            "recall_at_5": at(5), "recall_at_20": at(20)}


def resolve_targets(chapter_ranked, span_candidates, targets):
    """Resolve every labelled source for one query chapter.

    Each target keeps its own outcome: whether its chapter was retrieved by each
    rule, and whether the matched evidence touches the annotated source span and
    the annotated query span.
    """
    resolved = []
    for target in targets:
        book, chapter = parse_ref(target["ref"])
        chapter_rank = next(
            (rank for rank, row in enumerate(chapter_ranked, 1)
             if (row["book"], row["chapter"]) == (book, chapter)),
            None,
        )
        candidate = span_candidates.get((book, chapter))
        resolved.append({
            "target": target,
            "chapter_rank": chapter_rank,
            "span_rank": candidate["rank"] if candidate else None,
            "span_hit": span_hit(candidate, target["span"]) if candidate else False,
            "query_hit": query_span_hit(candidate, target["query_span"]) if candidate else False,
            "candidate": candidate,
        })
    return resolved


def summarise_targets(resolved):
    """Four distinct outcomes, so a high rank alone can never earn credit.

    span credit needs a span match; localization credit needs both a span match
    and the query-side verses doing the quoting. The primary target is reported
    separately so a secondary source cannot quietly stand in for it.
    """
    def minimum(predicate):
        ranks = [row["span_rank"] for row in resolved
                 if row["span_rank"] is not None and predicate(row)]
        return min(ranks) if ranks else None

    credited = sorted((row for row in resolved if row["span_hit"]),
                      key=lambda row: row["span_rank"])
    primary = next((row for row in resolved if row["target"]["primary"]), None)
    chapter_ranks = [row["chapter_rank"] for row in resolved
                     if row["chapter_rank"] is not None]

    return {
        "chapter_rank": min(chapter_ranks) if chapter_ranks else None,
        "span_rank": minimum(lambda row: row["span_hit"]),
        "localized_rank": minimum(lambda row: row["span_hit"] and row["query_hit"]),
        "credited": credited[0] if credited else None,
        "primary_chapter_rank": primary["chapter_rank"] if primary else None,
        "primary_span_rank": minimum(
            lambda row: primary is not None and row is primary and row["span_hit"]),
        "primary_hit": primary["span_hit"] if primary else None,
        "primary_query_hit": primary["query_hit"] if primary else None,
    }


def primary_credited_by_span(row):
    """Primary threshold credit: primary reference AND its annotated span."""
    top = row.get("top1")
    intervals = row.get("top1_intervals") or []
    for target in row.get("expected_targets", []):
        if (target.get("primary") and target["ref"] == top
                and span_hit({"source_intervals": intervals}, target["span"])):
            return True
    return False


def credited_by_span(row):
    """Was the top hit a labelled source whose matched range touches its span?"""
    top = row.get("top1")
    intervals = row.get("top1_intervals") or []
    for target in row.get("expected_targets", []):
        if target["ref"] == top and span_hit({"source_intervals": intervals}, target["span"]):
            return True
    return False


def threshold_metrics(rows, threshold):
    """Auto-accept every pair whose top hit scores at or above the threshold."""
    scored = [row for row in rows if row.get("top1_best") is not None]
    if not scored:
        return None

    accepted = [row for row in scored if row["top1_best"] >= threshold]
    # a top hit earns credit only if it is a labelled source AND its matched
    # evidence touches that source's annotated span
    true_positives = [row for row in accepted
                      if row["kind"] in RETRIEVAL_KINDS and credited_by_span(row)]
    primary_matches = [row for row in accepted
                       if row["kind"] in RETRIEVAL_KINDS
                       and primary_credited_by_span(row)]
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
        "true_positives_primary": len(primary_matches),
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


SPAN_TIERS = (
    ("chapter retrieved", "chapter_rank"),
    ("source span", "span_rank"),
    ("span + query spot", "localized_rank"),
    ("primary source span", "primary_span_rank"),
)
CHAPTER_TIERS = (
    ("chapter retrieved", "chapter_rank"),
    ("primary chapter", "primary_chapter_rank"),
)


def print_recall(title, rows, tiers):
    """One line per outcome tier, then per-kind at the source-span tier."""
    print(f"\n{title}")
    for label, rank_field in tiers:
        overall = recall_metrics(rows, rank_field)
        if not overall:
            continue
        print(f"  {label:<24} n={overall['pairs']:<3} "
              f"@1={overall['recall_at_1']:.0%}  @5={overall['recall_at_5']:.0%}  "
              f"@20={overall['recall_at_20']:.0%}")

    named = [row for row in rows if row["kind"] in ("quotation", "partial_quotation")]
    named_metrics = recall_metrics(named, "span_rank")
    if named_metrics:
        print(f"  {'quotations+partials':<24} n={named_metrics['pairs']:<3} "
              f"@1={named_metrics['recall_at_1']:.0%}  @5={named_metrics['recall_at_5']:.0%}  "
              f"@20={named_metrics['recall_at_20']:.0%}")

    for kind in RETRIEVAL_KINDS:
        subset = [row for row in rows if row["kind"] == kind]
        metrics = recall_metrics(subset, "span_rank")
        if not metrics:
            continue
        print(f"  {kind:<24} n={metrics['pairs']:<3} "
              f"@1={metrics['recall_at_1']:.0%}  @5={metrics['recall_at_5']:.0%}  "
              f"@20={metrics['recall_at_20']:.0%}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", default="data/bom.txt")
    parser.add_argument("--kjv", default="data/kjv.txt")
    parser.add_argument("--cache-index", default=".eval_cache",
                        help="directory for cached embeddings/meta (default .eval_cache)")
    parser.add_argument("--neighbors", type=int, default=50,
                        help="neighbours per query window (default 50)")
    parser.add_argument("--ranking", choices=["span", "chapter"], default="span",
                        help="rule used for the headline metrics (default span)")
    parser.add_argument("--tie-tolerance", type=float, default=0.01,
                        help="similarity band treated as a tie before support counts "
                             "(default 0.01). Dev cannot discriminate this value: every "
                             "dev positive is near-verbatim and ranks first at any "
                             "tolerance up to 0.02.")
    parser.add_argument("--json", help="write the full result as JSON here")
    args = parser.parse_args(argv)

    meta, embeddings = load_or_build_index(args)

    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    chapter_sizes = verses_per_chapter(meta)
    pairs = all_pairs()
    reviewed = sum(1 for pair in pairs if pair["reviewed"])
    results = []

    print(f"\nEvaluating {len(pairs)} labelled pairs against {len(meta)} windows "
          f"({reviewed} reviewed against the text, {len(pairs) - reviewed} not)")
    print(f"{'#':>3}  {'query':<12} {'kind':<18} {'split':<5} {'ch#':>4} {'sp#':>4} "
          f"{'loc#':>4} {'best':>7} {'sup':>3} {'src range':<10} {'span':<4}  "
          f"{'top-1 (span rule)':<18}")

    for number, pair in enumerate(pairs, 1):
        query_book, query_chapter = parse_ref(pair["query"])
        query_indices = [
            i for i, window in enumerate(meta)
            if window["corpus"] == "bom"
            and window["book"] == query_book
            and window["chapter"] == query_chapter
        ]
        if not query_indices:
            raise SystemExit(f"no windows for {pair['query']}")

        matches = collect_matches(meta, index, embeddings, query_indices, args.neighbors)
        chapter_ranked = rank_chapters(chapter_hits(meta, matches))
        span_candidates = aggregate_candidates(meta, matches, chapter_sizes)
        span_ranked = rank_candidates(span_candidates, args.tie_tolerance)

        expected = pair["expected"]
        resolved = resolve_targets(chapter_ranked, span_candidates,
                                   pair["expected_targets"])
        summary = summarise_targets(resolved)
        chapter_rank = summary["chapter_rank"]
        span_rank = summary["span_rank"]
        localized_rank = summary["localized_rank"]
        primary_chapter_rank = summary["primary_chapter_rank"]
        primary_span_rank = summary["primary_span_rank"]
        primary_hit = summary["primary_hit"]
        credited = summary["credited"]
        span_row = credited["candidate"] if credited else None
        hit_target = credited["target"] if credited else None
        source_hit = credited["span_hit"] if credited else None
        query_hit = credited["query_hit"] if credited else None
        credited_ref = credited["target"]["ref"] if credited else pair["expected"]
        chapter_row = None
        if credited_ref:
            credited_book, credited_chapter = parse_ref(credited_ref)
            chapter_row = next((row for row in chapter_ranked
                                if (row["book"], row["chapter"]) ==
                                (credited_book, credited_chapter)), None)

        top = span_ranked[0] if span_ranked else None
        top1 = f"{top['book']} {top['chapter']}" if top else None

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
            "expected_targets": pair["expected_targets"],
            "chapter_rank": chapter_rank,
            "chapter_best": chapter_row["best"] if chapter_row else None,
            "primary_chapter_rank": primary_chapter_rank,
            "primary_span_rank": primary_span_rank,
            "primary_span_hit": primary_hit,
            "localized_rank": localized_rank,
            "localized": localized_rank is not None,
            "hit_target": hit_target["ref"] if hit_target else None,
            "span_rank": span_rank,
            "span_best": span_row["best"] if span_row else None,
            "span_support": span_row["support"] if span_row else 0,
            "span_coverage": span_row["coverage"] if span_row else None,
            "source_range": span_row["source_range"] if span_row else None,
            "source_pairs": span_row["pairs"] if span_row else [],
            "source_pairs_total": len(span_row["pairs"]) if span_row else 0,
            "top1_intervals": (span_ranked[0].get("source_intervals") or []) if span_ranked else [],
            "query_range": span_row["query_range"] if span_row else None,
            "source_span_hit": source_hit,
            "query_span_hit": query_hit,
            "top1": top1,
            "top1_best": top["best"] if top else None,
            "top1_support": top["support"] if top else None,
            "candidates": len(span_ranked),
        })

        best_text = f"{span_row['best']:.4f}" if span_row else "-"
        span_flag = "yes" if source_hit else ("no" if source_hit is False else "-")
        credit = ""
        if hit_target and hit_target["ref"] != pair["expected"]:
            credit = f"  [credited: {hit_target['ref']}]"
        print(f"{number:>3}  {pair['query']:<12} {pair['kind']:<18} {pair['split']:<5} "
              f"{str(chapter_rank or '-'):>4} {str(span_rank or '-'):>4} "
              f"{str(localized_rank or '-'):>4} {best_text:>7} "
              f"{(span_row['support'] if span_row else 0):>3} "
              f"{(span_row['source_range'] if span_row else '-'):<10} {span_flag:<4}  "
              f"{(top1 or '-'):<18}{credit}")

    dev = [row for row in results if row["split"] == "dev"]
    test = [row for row in results if row["split"] == "test"]

    print_recall("Recall — dev split (tuning data) — chapter ranking", dev, CHAPTER_TIERS)
    print_recall("Recall — dev split (tuning data) — span ranking", dev, SPAN_TIERS)
    print_recall("Recall — test split (held out) — chapter ranking", test, CHAPTER_TIERS)
    print_recall("Recall — test split (held out) — span ranking", test, SPAN_TIERS)

    print("\nRegression cases (chapter rank -> span rank)")
    by_query = {row["query"]: row for row in results}
    for query in REGRESSION_CASES:
        row = by_query.get(query)
        if not row:
            continue
        span_flag = "yes" if row["source_span_hit"] else (
            "no" if row["source_span_hit"] is False else "-")
        best = f"{row['span_best']:.4f}" if row["span_best"] else "-"
        print(f"  {query:<12} {str(row['chapter_rank']):>4} -> {str(row['span_rank']):>4}  "
              f"best {best}  support {row['span_support']:>2}  "
              f"src {row['source_range'] or '-'}  span-hit {span_flag}")

    context_rows = [row for row in results if row["kind"] in CONTEXT_KINDS]
    if context_rows:
        print("\nContextual dependence, scored separately (not in recall)")
        for row in context_rows:
            if row["span_best"] is None:
                print(f"  {row['query']:<12} -> {row['expected']:<12} not retrieved")
            else:
                print(f"  {row['query']:<12} -> {row['expected']:<12} "
                      f"chapter rank {row['chapter_rank'] or '-'} | span rank "
                      f"{row['span_rank'] or '-'} | best {row['span_best']:.4f}")

    sweep_rows = threshold_sweep([row for row in dev if row["top1_best"] is not None])
    if sweep_rows:
        print(f"\nThreshold sweep on DEV only ({len(sweep_rows)} points)")
        print(f"  {'thresh':>6} {'accept':>6} {'tp':>4} {'tpP':>4} {'fpC':>4} {'ctx':>4} "
              f"{'other':>5} {'prec':>5} {'rec':>5} {'fpr':>5}")
        for row in sweep_rows:
            if row["accepted"] == 0:
                continue
            print(f"  {row['threshold']:>6.2f} {row['accepted']:>6} "
                  f"{row['true_positives']:>4} {row['true_positives_primary']:>4} "
                  f"{row['false_positives_controls']:>4} "
                  f"{row['accepted_context']:>4} {row['false_positives_other']:>5} "
                  f"{row['precision']:>5.2f} {row['recall']:>5.2f} "
                  f"{row['false_positive_rate']:>5.2f}")

    held_out = threshold_metrics([row for row in test if row["top1_best"] is not None],
                                 HELD_OUT_THRESHOLD)
    if held_out:
        print(f"\nHeld-out TEST at the threshold fixed in advance "
              f"({HELD_OUT_THRESHOLD:.2f}):")
        print(f"  accepted {held_out['accepted']} | tp {held_out['true_positives']} "
              f"(primary {held_out['true_positives_primary']}) | "
              f"fp-controls {held_out['false_positives_controls']} | "
              f"accepted-context {held_out['accepted_context']} | precision "
              f"{held_out['precision']:.2f} | recall {held_out['recall']:.2f} | fpr "
              f"{held_out['false_positive_rate']:.2f}")

    control_rows = [row for row in results if row["kind"] in CONTROL_KINDS]
    if control_rows:
        print("\nControls (what came back anyway)")
        for row in control_rows:
            flag = "" if row["reviewed"] else "  [unreviewed]"
            best = f"{row['top1_best']:.4f}" if row["top1_best"] is not None else "-"
            print(f"  {row['query']:<12} [{row['kind']:<17}] top-1 "
                  f"{(row['top1'] or '-'):<17} best {best}{flag}")

    if args.json:
        payload = {
            "windows": len(meta),
            "neighbors": args.neighbors,
            "window_size": WINDOW_SIZE,
            "window_overlap": WINDOW_OVERLAP,
            "min_document_frequency": MIN_DOCUMENT_FREQUENCY,
            "held_out_threshold": HELD_OUT_THRESHOLD,
            "ranking": args.ranking,
            "tie_tolerance": args.tie_tolerance,
            "labels_reviewed": reviewed,
            "labels_total": len(pairs),
            "results": results,
            "recall": {
                rule: {
                    "dev": recall_metrics(dev, f"{rule}_rank"),
                    "test": recall_metrics(test, f"{rule}_rank"),
                    "dev_span": span_recall_metrics(dev, f"{rule}_rank"),
                    "test_span": span_recall_metrics(test, f"{rule}_rank"),
                    "dev_localization": localization_metrics(dev, f"{rule}_rank"),
                    "test_localization": localization_metrics(test, f"{rule}_rank"),
                }
                for rule in ("chapter", "span")
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
