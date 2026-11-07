"""Classify the ranked passage pairs, walking them in score order.

`main.py` answers "these two passages look alike". `classifier.py` answers "what
kind of relationship is that, and how sure are we" — but until now nothing fed
it the pair list; it was only driven by the labelled evaluation harness.

This script is that bridge. It reads a pair CSV (default: top_matches_pairs.csv),
takes rows in descending cosine order, collapses the overlapping-window rows that
describe the same verse-span pair (a 30-token window grid reports one passage
pair many times), builds a classifier case per pair, and journals every answer to
JSONL so a long walk survives interruption.

Orientation: the canonically EARLIER side is presented as passage B (the source),
the later side as A (the query that may be quoting), because within a canon the
later text is the one doing the quoting. Each row keeps its own direction, so a
pair is classified once, not twice.

Usage:
    python classify_pairs.py --dry-run                  # selection counts + first prompt
    QI_LLM_URL=http://host:8000/v1/chat/completions \
    QI_LLM_MODEL=my-model \
    python classify_pairs.py --min-similarity 0.80 --limit 50
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from classifier import (
    RUBRIC_VERSION,
    ClassifierResponseError,
    RelationshipClassifier,
    build_case,
    http_completer,
)
from classify_evaluation import context_texts, verse_texts
from corpora import books_for, format_reference
from coverage import extent_reading, measure
from preprocess import preprocess_corpus

DEFAULT_MIN_SIMILARITY = 0.80
DEFAULT_CONTEXT_RADIUS = 2


def load_verse_texts(*specs):
    """(corpus, book, chapter, verse) -> verse text, for one or more corpora."""
    texts = {}
    for corpus_id, path in specs:
        for verse in preprocess_corpus(corpus_id, str(path)):
            texts[(verse["corpus"], verse["book"], verse["chapter"], verse["verse"])] = verse["text"]
    return texts


def book_order(corpus_id):
    """Canonical position of every book, so 'earlier' is well defined."""
    return {book: index for index, book in enumerate(books_for(corpus_id))}


def _sort_key(book, chapter, verse, order):
    return (order.get(book, len(order)), chapter, verse)


def orient(row, orders, source_corpus=None):
    """Decide which side is the source (B) and which is the query (A).

    Within one corpus the canonically earlier side is the source: the later text
    is the one doing the quoting. Across corpora canonical position means
    nothing, so the caller names the corpus that plays the source role — for
    Book of Mormon against King James, the Bible is always the source.
    """
    corpus_1, corpus_2 = row["corpus_1"], row["corpus_2"]

    if corpus_1 != corpus_2:
        if source_corpus is None:
            raise ValueError(
                f"cross-corpus pair {corpus_1}/{corpus_2} needs a source corpus"
            )
        if source_corpus not in (corpus_1, corpus_2):
            raise ValueError(
                f"source corpus {source_corpus!r} is not in the pair "
                f"{corpus_1}/{corpus_2}"
            )
        earlier, later = ("1", "2") if corpus_1 == source_corpus else ("2", "1")
    else:
        order = orders[corpus_1]
        side_1 = _sort_key(row["book_1"], row["chapter_1"], row["verse_1"], order)
        side_2 = _sort_key(row["book_2"], row["chapter_2"], row["verse_2"], order)
        earlier, later = ("1", "2") if side_1 <= side_2 else ("2", "1")

    return {
        "query_corpus": row[f"corpus_{later}"],
        "source_corpus": row[f"corpus_{earlier}"],
        "query_book": row[f"book_{later}"],
        "query_chapter": int(row[f"chapter_{later}"]),
        "query_verse": int(row[f"verse_{later}"]),
        "query_end_verse": int(row[f"end_verse_{later}"]),
        "source_book": row[f"book_{earlier}"],
        "source_chapter": int(row[f"chapter_{earlier}"]),
        "source_verse": int(row[f"verse_{earlier}"]),
        "source_end_verse": int(row[f"end_verse_{earlier}"]),
        "query_rank": row[f"rank_{later}"],
        "source_rank": row[f"rank_{earlier}"],
        "cosine_similarity": float(row["cosine_similarity"]),
    }


def pair_key(oriented):
    return (
        oriented["query_corpus"],
        oriented["query_book"], oriented["query_chapter"],
        oriented["query_verse"], oriented["query_end_verse"],
        oriented["source_corpus"],
        oriented["source_book"], oriented["source_chapter"],
        oriented["source_verse"], oriented["source_end_verse"],
    )


def key_text(oriented):
    """Stable human-readable key, used for journal resume matching.

    The corpus is named only when the two sides come from different corpora, so
    keys written by an earlier within-corpus run still match.
    """
    query = format_reference(
        oriented["query_book"], oriented["query_chapter"],
        oriented["query_verse"], oriented["query_end_verse"],
    )
    source = format_reference(
        oriented["source_book"], oriented["source_chapter"],
        oriented["source_verse"], oriented["source_end_verse"],
    )
    if oriented["query_corpus"] == oriented["source_corpus"]:
        return f"{query} -> {source}"
    return f"{query} [{oriented['query_corpus']}] -> {source} [{oriented['source_corpus']}]"


def collapse_span_pairs(frame, orders, source_corpus=None):
    """One record per verse-span pair, keeping the strongest hit and all windows.

    The window grid reports the same passage pair once per overlapping window;
    classifying each of those would ask the model the same question repeatedly.
    Every supporting window is kept as evidence, so nothing about the retrieval
    signal is lost — only the repeated model call is.
    """
    merged = {}

    for row in frame.to_dict("records"):
        oriented = orient(row, orders, source_corpus)
        key = pair_key(oriented)
        entry = merged.get(key)

        if entry is None:
            entry = merged[key] = {
                "oriented": oriented,
                "key_text": key_text(oriented),
                "cosine_similarity": oriented["cosine_similarity"],
                "windows": [],
            }
        elif oriented["cosine_similarity"] > entry["cosine_similarity"]:
            # keep the strongest row's ranks, but retain every window below
            entry["oriented"] = oriented
            entry["cosine_similarity"] = oriented["cosine_similarity"]

        entry["windows"].append({
            "query_range": format_reference(
                oriented["query_book"], oriented["query_chapter"],
                oriented["query_verse"], oriented["query_end_verse"],
            ),
            "source_range": format_reference(
                oriented["source_book"], oriented["source_chapter"],
                oriented["source_verse"], oriented["source_end_verse"],
            ),
            "similarity": oriented["cosine_similarity"],
        })

    return sorted(merged.values(), key=lambda entry: -entry["cosine_similarity"])


def span_verses(texts, corpus_id, book, chapter, start, end):
    return [(verse, texts[(corpus_id, book, chapter, verse)])
            for verse in range(start, end + 1)
            if (corpus_id, book, chapter, verse) in texts]


def build_pair_case(entry, texts, context_radius=DEFAULT_CONTEXT_RADIUS):
    """Build the classifier input for one verse-span pair.

    Each side reads from its own corpus, so a cross-corpus pair is the same code
    path as a within-corpus one.
    """
    oriented = entry["oriented"]
    query_corpus = oriented["query_corpus"]
    source_corpus = oriented["source_corpus"]
    query_ref = format_reference(
        oriented["query_book"], oriented["query_chapter"],
        oriented["query_verse"], oriented["query_end_verse"],
    )
    source_ref = format_reference(
        oriented["source_book"], oriented["source_chapter"],
        oriented["source_verse"], oriented["source_end_verse"],
    )

    query_text = verse_texts(
        texts, query_corpus, oriented["query_book"], oriented["query_chapter"],
        oriented["query_verse"], oriented["query_end_verse"],
    )
    source_text = verse_texts(
        texts, source_corpus, oriented["source_book"], oriented["source_chapter"],
        oriented["source_verse"], oriented["source_end_verse"],
    )

    source_verses = span_verses(
        texts, source_corpus, oriented["source_book"], oriented["source_chapter"],
        oriented["source_verse"], oriented["source_end_verse"],
    )
    reproduction = measure(source_verses, query_text)
    reading = extent_reading(reproduction)
    reproduction["extent"] = reading["extent"]
    reproduction["extent_reason"] = reading["reason"]

    return build_case(
        query={
            "ref": query_ref,
            "span": f"{oriented['query_verse']}-{oriented['query_end_verse']}",
            "text": query_text,
            "context": context_texts(
                texts, query_corpus, oriented["query_book"], oriented["query_chapter"],
                oriented["query_verse"], oriented["query_end_verse"],
                radius=context_radius,
            ),
        },
        source={
            "ref": source_ref,
            "span": f"{oriented['source_verse']}-{oriented['source_end_verse']}",
            "text": source_text,
            "context": context_texts(
                texts, source_corpus, oriented["source_book"], oriented["source_chapter"],
                oriented["source_verse"], oriented["source_end_verse"],
                radius=context_radius,
            ),
        },
        matched_windows=entry["windows"],
        retrieval={
            "best": entry["cosine_similarity"],
            "support": len(entry["windows"]),
            "coverage": None,
        },
        reproduction=reproduction,
    )


OUTPUT_COLUMNS = [
    "cosine_similarity", "query_ref", "query_corpus", "source_ref", "source_corpus",
    "query_span", "source_span",
    "relationship", "secondary_relationship", "confidence", "needs_review",
    "review_reasons", "classifier_query_span", "classifier_source_span",
    "guard_applied", "guard_original_relationship", "reproduced_share",
    "extent_reading", "explanation", "evidence", "model", "rubric_version",
    "attempts", "error",
]


def result_row(entry, result=None, error=None, attempts=1):
    """Flatten one classification (or its failure) into an output row."""
    oriented = entry["oriented"]
    row = {
        "cosine_similarity": round(entry["cosine_similarity"], 4),
        "query_ref": format_reference(
            oriented["query_book"], oriented["query_chapter"],
            oriented["query_verse"], oriented["query_end_verse"],
        ),
        "query_corpus": oriented["query_corpus"],
        "source_ref": format_reference(
            oriented["source_book"], oriented["source_chapter"],
            oriented["source_verse"], oriented["source_end_verse"],
        ),
        "source_corpus": oriented["source_corpus"],
        "query_span": f"{oriented['query_verse']}-{oriented['query_end_verse']}",
        "source_span": f"{oriented['source_verse']}-{oriented['source_end_verse']}",
        "attempts": attempts,
    }

    if error is not None:
        # Blank, not empty strings: an unclassified pair must not turn the numeric
        # columns into text for every consumer that reads this file.
        row.update({column: None for column in OUTPUT_COLUMNS if column not in row})
        row["error"] = str(error)
        return row

    reproduction = result.get("reproduction") or {}
    row.update({
        "relationship": result["relationship"],
        "secondary_relationship": result.get("secondary_relationship", ""),
        "confidence": result["confidence"],
        "needs_review": bool(result["needs_review"]),
        "review_reasons": "; ".join(result.get("review_reasons", [])),
        "classifier_query_span": result.get("query_span", ""),
        "classifier_source_span": result.get("source_span", ""),
        "guard_applied": result.get("guard_applied", ""),
        "guard_original_relationship": result.get("guard_original_relationship", ""),
        "reproduced_share": reproduction.get("reproduced_share"),
        "extent_reading": reproduction.get("extent", ""),
        "explanation": result.get("explanation", ""),
        "evidence": " | ".join(result.get("evidence", [])),
        "model": result.get("model", ""),
        "rubric_version": result.get("rubric_version", ""),
        "error": "",
    })
    return row


def read_journal(path):
    """key -> journal record, last write wins."""
    records = {}
    path = Path(path)
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated final line from a killed run
        records[record["key"]] = record
    return records


def append_journal(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def select(entries, min_similarity, limit):
    chosen = [entry for entry in entries
              if entry["cosine_similarity"] >= min_similarity]
    if limit:
        chosen = chosen[:limit]
    return chosen


def write_output(records, output_path):
    rows = [record["row"] for record in records]
    frame = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    frame.to_csv(output_path, index=False)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="top_matches_pairs.csv",
                        help="pair CSV from main.py (default: top_matches_pairs.csv)")
    parser.add_argument("--output", default="classifications.csv")
    parser.add_argument("--journal", default="classifications.jsonl")
    parser.add_argument("--corpus", default="data/bom.txt")
    parser.add_argument("--corpus-id", default="bom")
    parser.add_argument("--source-corpus", default=None,
                        help="second corpus file, for a cross-corpus pair list; it is "
                             "always presented as the SOURCE side")
    parser.add_argument("--source-corpus-id", default=None,
                        help="parser for --source-corpus (default: --corpus-id)")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY,
                        help=f"skip pairs below this cosine (default: {DEFAULT_MIN_SIMILARITY})")
    parser.add_argument("--limit", type=int, default=0,
                        help="classify at most N pairs after filtering (default: all)")
    parser.add_argument("--context-radius", type=int, default=DEFAULT_CONTEXT_RADIUS)
    parser.add_argument("--llm-url", default=None, help="default: $QI_LLM_URL")
    parser.add_argument("--llm-model", default=None, help="default: $QI_LLM_MODEL")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--restart", action="store_true",
                        help="ignore the existing journal and start over")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selection counts and the first prompt; no model calls")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.input)

    source_id = args.source_corpus_id or args.corpus_id
    corpora_in_file = set(frame["corpus_1"].dropna()) | set(frame["corpus_2"].dropna())

    if len(corpora_in_file) > 1 and not args.source_corpus_id:
        parser.exit(
            2,
            f"{parser.prog}: error: this pair list mixes corpora "
            f"({', '.join(sorted(corpora_in_file))}); say which one is the source "
            f"with --source-corpus-id\n",
        )

    specs = [(args.corpus_id, args.corpus)]
    if source_id != args.corpus_id:
        if not args.source_corpus:
            parser.exit(
                2,
                f"{parser.prog}: error: --source-corpus-id {source_id} needs "
                f"--source-corpus PATH\n",
            )
        specs.append((source_id, args.source_corpus))

    orders = {corpus_id: book_order(corpus_id) for corpus_id, _ in specs}
    entries = collapse_span_pairs(frame, orders, source_corpus=source_id)
    chosen = select(entries, args.min_similarity, args.limit)

    texts = load_verse_texts(*specs)

    journal_path = Path(args.journal)
    if args.restart and journal_path.exists():
        journal_path.unlink()
    done = read_journal(journal_path)

    def succeeded(record):
        """A journal record counts only if it produced a classification."""
        return bool(record) and not str(record.get("row", {}).get("error") or "").strip()

    pending = [entry for entry in chosen if not succeeded(done.get(entry["key_text"]))]
    failed_before = sum(
        1 for entry in chosen
        if entry["key_text"] in done and not succeeded(done[entry["key_text"]])
    )
    already = len(chosen) - len(pending)

    print(f"pair rows in {args.input}: {len(frame)}")
    print(f"distinct verse-span pairs: {len(entries)}")
    print(f"selected at cosine >= {args.min_similarity}: {len(chosen)}")
    if already:
        print(f"already classified (resuming): {already}")
    if failed_before:
        print(f"previously failed, retrying: {failed_before}")
    print(f"model calls to make: {len(pending)}")

    if args.dry_run:
        if not chosen:
            print("nothing selected", file=sys.stderr)
            return 1
        entry = chosen[0]
        case = build_pair_case(entry, texts, args.context_radius)
        from classifier import SYSTEM_PROMPT, render_prompt
        print("\n--- first case: " + entry["key_text"] + " ---")
        print(SYSTEM_PROMPT)
        print()
        print(render_prompt(case))
        return 0

    if not pending:
        records = list(done.values())
        write_output(records, args.output)
        print(f"nothing to do; wrote {len(records)} rows to {args.output}")
        return 0

    complete = http_completer(url=args.llm_url, model=args.llm_model,
                              timeout=args.timeout, thinking=args.thinking)
    classifier = RelationshipClassifier(complete, model=args.llm_model or None)

    errors = 0
    repaired = 0
    for position, entry in enumerate(pending, start=1):
        case = build_pair_case(entry, texts, args.context_radius)
        attempts = 1
        try:
            try:
                result = classifier.classify(case)
            except ClassifierResponseError as parse_error:
                # The answer arrived but could not be used. One repair call is
                # cheaper than losing the pair; a second malformed answer is
                # journaled as an error and retried on the next run.
                attempts = 2
                repaired += 1
                result = classifier.classify(case, repair_note=str(parse_error))
            row = result_row(entry, result, attempts=attempts)
            record = {"key": entry["key_text"], "row": row}
            note = (f"{result['relationship']:<18} conf {result['confidence']:.2f} "
                    f"{'REVIEW' if result['needs_review'] else '      '}")
        except Exception as error:  # noqa: BLE001 - one bad answer must not end the walk
            errors += 1
            row = result_row(entry, error=error, attempts=attempts)
            record = {"key": entry["key_text"], "row": row}
            note = f"ERROR {error}"

        append_journal(journal_path, record)
        done[entry["key_text"]] = record
        print(f"[{position}/{len(pending)}] {entry['key_text']:<28} "
              f"cos {entry['cosine_similarity']:.3f}  {note}")

    records = sorted(done.values(), key=lambda record: -record["row"]["cosine_similarity"])
    frame_out = write_output(records, args.output)
    print(f"\n{len(frame_out)} rows -> {args.output}  "
          f"({errors} errors, {repaired} retried after a malformed answer)")

    if not frame_out.empty:
        counts = frame_out["relationship"].fillna("ERROR").replace("", "ERROR").value_counts()
        print("\nrelationship counts:")
        for label, count in counts.items():
            print(f"  {label:<20} {count:>5}")
        reviewed = int(frame_out["needs_review"].fillna(False).astype(bool).sum())
        print(f"  flagged for review  {reviewed:>5}")
    print(f"rubric version {RUBRIC_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
