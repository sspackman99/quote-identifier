"""Run the classifier over the labelled evaluation set and score it against the labels.

This is the bridge between retrieval and classification: it uses the SAME cached
index and the SAME candidate ranking as evaluate_retrieval.py, takes the top-ranked
candidate a user would actually be shown, classifies it, and then compares the
classifier's answer with the human label.

Retrieval is untouched: nothing here changes scores, thresholds or spans.

Scoring is reported on two populations, because they answer different questions:

  all pairs            end-to-end: retrieval plus classification
  retrieval-correct    classification only, on pairs where the top candidate really
                       is a labelled source (the classifier cannot be blamed for a
                       retrieval miss, and a good classifier cannot rescue one)

Usage:
    python classify_evaluation.py --limit 4 --dry-run          # print the first prompt
    QI_LLM_URL=http://10.77.1.117:8000/v1/chat/completions \
    QI_LLM_MODEL=qwen3.5-9b python classify_evaluation.py --json classification.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from candidates import (
    aggregate_candidates,
    collect_matches,
    expand_span,
    rank_candidates,
    verses_per_chapter,
)
from classifier import RELATIONSHIPS, RelationshipClassifier, build_case, http_completer
from evaluation_pairs import RETRIEVAL_KINDS, all_pairs
from evaluate_retrieval import load_or_build_index, parse_ref, resolve_targets, summarise_targets
from preprocess import preprocess_corpus

CONTEXT_RADIUS = 2
NEIGHBOURS = 50
TIE_TOLERANCE = 0.01


def load_bundle(args):
    import faiss
    import numpy as np

    namespace = argparse.Namespace(bom=args.bom, kjv=args.kjv,
                                   cache_index=args.cache_index)
    meta, embeddings = load_or_build_index(namespace)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(float32(embeddings))
    return meta, embeddings, index, verses_per_chapter(meta)


def float32(embeddings):
    import numpy as np
    return np.asarray(embeddings, dtype="float32")


def load_texts(bom_path, kjv_path):
    texts = {}
    for corpus_id, path in (("bom", bom_path), ("kjv", kjv_path)):
        for verse in preprocess_corpus(corpus_id, str(path)):
            texts[(corpus_id, verse["book"], verse["chapter"], verse["verse"])] = verse["text"]
    return texts


def verse_texts(texts, corpus, book, chapter, start, end):
    lines = []
    for verse in range(start, end + 1):
        text = texts.get((corpus, book, chapter, verse))
        if text:
            lines.append(f"{chapter}:{verse} {text}")
    return "\n".join(lines)


def context_texts(texts, corpus, book, chapter, start, end, radius=CONTEXT_RADIUS):
    available = [verse for (c, b, ch, verse) in texts
                 if c == corpus and b == book and ch == chapter]
    if not available:
        return ""
    low = max(min(available), start - radius)
    high = min(max(available), end + radius)
    return verse_texts(texts, corpus, book, chapter, low, high)


def parse_span(text):
    """'7-10', '7', '7,10' -> a set of verse numbers. Empty or junk -> empty set."""
    numbers = set()
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            if low.isdigit() and high.isdigit():
                numbers |= set(range(int(low), int(high) + 1))
        elif part.isdigit():
            numbers.add(int(part))
    return numbers


def expected_label_for(pair):
    """The label to compare against: primary kind, or the credited target's kind."""
    return pair["kind"]


def candidate_for(meta, embeddings, index, sizes, pair, texts, mode="retrieved"):
    """Build the classifier input for one pair.

    mode="retrieved": the top-ranked candidate, i.e. what a user would actually see.
                      Errors here mix retrieval and classification.
    mode="expected":  the labelled source chapter, so the confusion matrix measures
                      classification alone. This is NOT what a user sees; it exists
                      to separate the two failure modes.
    """
    query_book, query_chapter = parse_ref(pair["query"])
    query_indices = [i for i, window in enumerate(meta)
                     if window["corpus"] == "bom"
                     and window["book"] == query_book
                     and window["chapter"] == query_chapter]
    if not query_indices:
        return None

    matches = collect_matches(meta, index, embeddings, query_indices, NEIGHBOURS)
    candidates = aggregate_candidates(meta, matches, sizes)
    ranked = rank_candidates(candidates, tie_tolerance=TIE_TOLERANCE)
    if not ranked:
        return None

    expected_refs = {target["ref"] for target in pair["expected_targets"]}
    resolved = resolve_targets([], candidates, pair["expected_targets"])
    summary = summarise_targets(resolved)

    chosen_span = None
    chosen_query_span = pair["query_span"]
    if mode == "expected":
        target = summary["credited"]["target"] if summary["credited"] else None
        if target is None:
            return None
        target_book, target_chapter = parse_ref(target["ref"])
        top = candidates.get((target_book, target_chapter))
        if top is None:
            return None
        chosen_span = target["span"]
        chosen_query_span = target.get("query_span") or pair["query_span"]
    else:
        top = ranked[0]

    query_span = chosen_query_span
    if query_span and query_span != "whole":
        query_verses = expand_span(query_span)
        query_start, query_end = min(query_verses), max(query_verses)
    else:
        # 'whole' means the chapter itself; use its full verse range
        chapter_verses = [window["start_verse"] for window in meta
                          if window["corpus"] == "bom"
                          and window["book"] == query_book
                          and window["chapter"] == query_chapter]
        query_start, query_end = min(chapter_verses), max(
            window["end_verse"] for window in meta
            if window["corpus"] == "bom" and window["book"] == query_book
            and window["chapter"] == query_chapter)

    if chosen_span and chosen_span != "whole":
        source_verses = expand_span(chosen_span)
        source_start, source_end = min(source_verses), max(source_verses)
    else:
        source_intervals = top["source_intervals"]
        source_start = min(start for start, _ in source_intervals)
        source_end = max(end for _, end in source_intervals)

    case = build_case(
        query={
            "ref": pair["query"],
            "span": chosen_query_span or "whole chapter",
            "text": verse_texts(texts, "bom", query_book, query_chapter,
                                query_start, query_end),
            "context": context_texts(texts, "bom", query_book, query_chapter,
                                     query_start, query_end),
        },
        source={
            "ref": f"{top['book']} {top['chapter']}",
            "span": top["source_range"],
            "text": verse_texts(texts, "kjv", top["book"], top["chapter"],
                                source_start, source_end),
            "context": context_texts(texts, "kjv", top["book"], top["chapter"],
                                     source_start, source_end),
        },
        matched_windows=[{"query_range": window["query_range"],
                          "source_range": window["source_range"],
                          "similarity": window["sim"]}
                         for window in top["pairs"]],
        retrieval={"best": top["best"], "support": top["support"],
                   "coverage": top["coverage"]},
    )

    return {
        "case": case,
        "candidate_ref": f"{top['book']} {top['chapter']}",
        "mode": mode,
        "rank": top["rank"],
        "source_range": top["source_range"],
        "best": top["best"],
        "retrieval_ok": f"{top['book']} {top['chapter']}" in expected_refs,
        "expected_target_span": (summary["credited"]["target"]["span"]
                                 if summary["credited"] else None),
        "expected_target_ref": (summary["credited"]["target"]["ref"]
                                if summary["credited"] else None),
    }


def span_agreement(predicted, expected):
    """Does the classifier's source span overlap the span it should have found?"""
    predicted_verses = parse_span(predicted)
    expected_verses = parse_span(expected)
    if not predicted_verses or not expected_verses:
        return None
    return bool(predicted_verses & expected_verses)


def confusion(rows):
    matrix = defaultdict(Counter)
    for row in rows:
        matrix[row["expected"]][row["predicted"] or "ERROR"] += 1
    return matrix


def print_confusion(matrix, labels):
    header = "expected \\ predicted".ljust(20) + "".join(f"{label[:11]:>13}" for label in labels)
    print(header)
    for expected in labels:
        if not matrix.get(expected):
            continue
        row = f"  {expected:<18}"
        for label in labels:
            row += f"{matrix[expected].get(label, 0):>13}"
        print(row)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", default="data/bom.txt")
    parser.add_argument("--kjv", default="data/kjv.txt")
    parser.add_argument("--cache-index", default=".eval_cache")
    parser.add_argument("--llm-url", default=None, help="chat completions endpoint "
                                                        "(default: $QI_LLM_URL)")
    parser.add_argument("--llm-model", default=None, help="model name (default: $QI_LLM_MODEL)")
    parser.add_argument("--candidate", choices=["retrieved", "expected"],
                        default="retrieved",
                        help="classify the top-ranked candidate (end to end) or the "
                             "labelled source chapter (classification only)")
    parser.add_argument("--limit", type=int, default=None,
                        help="classify only the first N pairs")
    parser.add_argument("--only", default=None,
                        help="comma-separated query references to classify")
    parser.add_argument("--timeout", type=int, default=None,
                        help="per-request seconds (default: $QI_LLM_TIMEOUT or 120)")
    parser.add_argument("--thinking", action="store_true",
                        help="let a reasoning model think before answering (slow)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the first rendered prompt and make no model calls")
    parser.add_argument("--json", default=None, help="write results here")
    args = parser.parse_args(argv)

    meta, embeddings, index, sizes = load_bundle(args)
    texts = load_texts(args.bom, args.kjv)

    pairs = all_pairs()
    if args.only:
        wanted = {part.strip() for part in args.only.split(",")}
        pairs = [pair for pair in pairs if pair["query"] in wanted]
    if args.limit:
        pairs = pairs[:args.limit]

    prepared = []
    for pair in pairs:
        candidate = candidate_for(meta, embeddings, index, sizes, pair, texts,
                                  mode=args.candidate)
        if candidate is None:
            print(f"  ! no candidate for {pair['query']}", file=sys.stderr)
            continue
        prepared.append((pair, candidate))

    if args.dry_run:
        pair, candidate = prepared[0]
        print(f"--- {pair['query']} -> {candidate['candidate_ref']} ---")
        from classifier import SYSTEM_PROMPT, render_prompt
        print(SYSTEM_PROMPT)
        print()
        print(render_prompt(candidate["case"]))
        return 0

    complete = http_completer(url=args.llm_url, model=args.llm_model,
                              timeout=args.timeout, thinking=args.thinking)
    classifier = RelationshipClassifier(complete, model=args.llm_model or None)

    rows = []
    print(f"{'query':<14} {'expected':<18} {'predicted':<18} {'conf':>5} {'review':<7} "
          f"{'ret':<4} {'candidate':<16}")
    for pair, candidate in prepared:
        expected = expected_label_for(pair)
        try:
            result = classifier.classify(candidate["case"])
        except Exception as error:  # noqa: BLE001 - one bad answer must not end the run
            rows.append({"query": pair["query"], "expected": expected,
                         "predicted": None, "confidence": None, "error": str(error),
                         "retrieval_ok": candidate["retrieval_ok"],
                         "candidate": candidate["candidate_ref"]})
            print(f"{pair['query']:<14} {expected:<18} {'ERROR':<18} "
                  f"{'-':>5} {'-':<7} {'y' if candidate['retrieval_ok'] else 'n':<4} "
                  f"{candidate['candidate_ref']:<16} {error}")
            continue

        agreement = span_agreement(result["source_span"], candidate["expected_target_span"])
        rows.append({
            "query": pair["query"],
            "expected": expected,
            "predicted": result["relationship"],
            "confidence": result["confidence"],
            "matched_windows_shown": len(result["matched_windows"]),
            "matched_windows_total": result["matched_total"],
            "needs_review": result["needs_review"],
            "review_reasons": result["review_reasons"],
            "explanation": result["explanation"],
            "evidence": result["evidence"],
            "classifier_query_span": result["query_span"],
            "classifier_source_span": result["source_span"],
            "span_agreement": agreement,
            "retrieval_ok": candidate["retrieval_ok"],
            "candidate": candidate["candidate_ref"],
            "candidate_span": candidate["source_range"],
            "expected_target_ref": candidate["expected_target_ref"],
            "expected_target_span": candidate["expected_target_span"],
            "similarity": candidate["best"],
        })
        print(f"{pair['query']:<14} {expected:<18} {result['relationship']:<18} "
              f"{result['confidence']:>5.2f} {'yes' if result['needs_review'] else '-':<7} "
              f"{'y' if candidate['retrieval_ok'] else 'n':<4} "
              f"{candidate['candidate_ref']:<16}")

    scored = [row for row in rows if row["predicted"]]
    correct = [row for row in scored if row["predicted"] == row["expected"]]
    classified = [row for row in scored if row["retrieval_ok"]]
    correct_classified = [row for row in classified if row["predicted"] == row["expected"]]

    print(f"\nclassified {len(scored)}/{len(rows)}")
    print(f"  end-to-end accuracy          {len(correct)}/{len(scored)}")
    print(f"  classification accuracy      {len(correct_classified)}/{len(classified)} "
          f"(retrieval-correct pairs only)")
    if scored:
        agreement = [row["span_agreement"] for row in scored
                     if row["span_agreement"] is not None]
        if agreement:
            print(f"  source-span agreement        {sum(agreement)}/{len(agreement)}")
        reviewed = sum(1 for row in scored if row["needs_review"])
        print(f"  flagged for review           {reviewed}/{len(scored)}")

    if classified:
        print("\nConfusion — retrieval-correct pairs only")
        print_confusion(confusion(classified), RELATIONSHIPS)

    mistakes = [row for row in classified if row["predicted"] != row["expected"]]
    if mistakes:
        print("\nMistakes on retrieval-correct pairs")
        for row in mistakes:
            print(f"  {row['query']:<14} expected {row['expected']:<18} "
                  f"got {row['predicted']:<18} conf {row['confidence']:.2f}")
            print(f"    {row['explanation'][:150]}")

    if args.json:
        payload = {
            "model": classifier.model,
            "pairs": len(rows),
            "classified": len(scored),
            "end_to_end_accuracy": f"{len(correct)}/{len(scored)}",
            "classification_accuracy": f"{len(correct_classified)}/{len(classified)}",
            "rows": rows,
            "confusion_retrieval_correct": {expected: dict(counter)
                                            for expected, counter
                                            in confusion(classified).items()},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nJSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
