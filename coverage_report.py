"""Measure verbatim reproduction for every labelled pair and compare with the labels.

This is the deterministic extent measurement, run over the evaluation set. It makes no
model calls and changes nothing: it reports what the text does, so the labels and the
classifier can be judged against a number rather than an impression.

Usage:
    python coverage_report.py --json coverage_report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from coverage import extent_reading, measure
from evaluation_pairs import all_pairs
from evaluate_retrieval import parse_ref
from preprocess import preprocess_corpus

CORPORA = {"bom": "bom", "kjv": "kjv"}


def load_corpora(bom_path, kjv_path):
    texts = {}
    for corpus_id, path in (("bom", bom_path), ("kjv", kjv_path)):
        for verse in preprocess_corpus(corpus_id, str(path)):
            texts[(corpus_id, verse["book"], verse["chapter"], verse["verse"])] = verse["text"]
    return texts


def chapter_verses(texts, corpus, book, chapter):
    rows = [(verse, text) for (c, b, ch, verse), text in texts.items()
            if c == corpus and b == book and ch == chapter]
    return sorted(rows)


def chapter_text(texts, corpus, book, chapter):
    return " ".join(text for _, text in chapter_verses(texts, corpus, book, chapter))


def parse_span_set(span):
    """'7-10' -> {7,8,9,10}; '7' -> {7}; anything else -> empty set."""
    numbers = set()
    if not span or span == "whole":
        return numbers
    for part in str(span).replace(" ", "").split(","):
        if "-" in part:
            low, _, high = part.partition("-")
            if low.isdigit() and high.isdigit():
                numbers |= set(range(int(low), int(high) + 1))
        elif part.isdigit():
            numbers.add(int(part))
    return numbers


def span_text(texts, corpus, book, chapter, span):
    verses = chapter_verses(texts, corpus, book, chapter)
    wanted = parse_span_set(span)
    if wanted:
        verses = [(number, text) for number, text in verses if number in wanted]
    return " ".join(text for _, text in verses)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bom", default="data/bom.txt")
    parser.add_argument("--kjv", default="data/kjv.txt")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    texts = load_corpora(args.bom, args.kjv)
    rows = []

    print(f"{'query':<14} {'expected':<13} {'label':<18} {'rep%':>5} {'run':>4} "
          f"{'query%':>7}  {'extent reading':<20} repro verses")
    for pair in all_pairs():
        targets = pair["expected_targets"]
        if not targets:
            continue
        target = targets[0]
        source_book, source_chapter = parse_ref(target["ref"])
        query_book, query_chapter = parse_ref(pair["query"])

        source_verses = chapter_verses(texts, "kjv", source_book, source_chapter)
        query_passage = chapter_text(texts, "bom", query_book, query_chapter)
        if not source_verses or not query_passage:
            continue

        measurement = measure(source_verses, query_passage)
        reading = extent_reading(measurement)
        # what the pair claims, measured inside the annotated span only
        wanted_verses = parse_span_set(target["span"])
        span_verses = [(number, text) for number, text in source_verses
                       if not wanted_verses or number in wanted_verses]
        span_measurement = measure(span_verses, query_passage) if span_verses else None

        row = {
            "query": pair["query"],
            "expected": target["ref"],
            "label": pair["kind"],
            "span": target["span"],
            "reproduced_share": measurement["reproduced_share"],
            "reproduced_count": measurement["reproduced_count"],
            "verses_in_source": measurement["verses_in_source"],
            "longest_run": measurement["longest_run"],
            "query_share": measurement["query_share"],
            "reproduced_verses": measurement["reproduced_verses"],
            "extent": reading["extent"],
            "extent_reason": reading["reason"],
            "span_reproduced_share": (span_measurement["reproduced_share"]
                                      if span_measurement else None),
        }
        rows.append(row)

        print(f"{pair['query']:<14} {target['ref']:<13} {pair['kind']:<18} "
              f"{measurement['reproduced_share']:>5.0%} {measurement['longest_run']:>4} "
              f"{measurement['query_share']:>7.0%}  {reading['extent']:<20} "
              f"{measurement['reproduced_verses']}")

    print("\nSeparation by label (reproduced share of the source chapter)")
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row["reproduced_share"])
    for label in sorted(by_label):
        shares = sorted(by_label[label])
        print(f"  {label:<18} n={len(shares):<2} min {shares[0]:.0%}  "
              f"median {shares[len(shares) // 2]:.0%}  max {shares[-1]:.0%}")

    quoted = [row for row in rows if row["label"] in ("quotation", "partial_quotation")]
    if quoted:
        correct = [row for row in quoted
                   if (row["label"] == "quotation" and row["extent"] == "quotation")
                   or (row["label"] == "partial_quotation"
                       and row["extent"] == "partial_quotation")]
        print(f"\nExtent reading agrees with the quotation/partial label on "
              f"{len(correct)}/{len(quoted)} pairs")
        for row in quoted:
            if row not in correct:
                print(f"  disagrees: {row['query']:<14} label {row['label']:<18} "
                      f"reading {row['extent']:<18} share {row['reproduced_share']:.0%}")

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(f"\nJSON written to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
