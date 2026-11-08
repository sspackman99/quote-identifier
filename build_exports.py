"""Build the downloadable CSV and the data file behind the network visualisation.

Two artefacts, both derived from the per-pair journals:

  matrix_classifications.csv   every classified pair in one table, with the book,
                               chapter and verse on each side parsed out of the
                               reference so it can be filtered without string work
  viz/data.json               book nodes, book-to-book edges and the underlying
                               pairs, ready for the static page in viz/

Run it after classify_matrix.py; it is cheap and idempotent.
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from corpora import load_corpus

CORPUS_LABELS = {
    "bom": "Book of Mormon",
    "ot": "Old Testament",
    "nt": "New Testament",
    "dc": "Doctrine and Covenants",
    "pgp": "Pearl of Great Price",
    "kjv": "Bible (KJV)",
}

CORPUS_ORDER = ["bom", "ot", "nt", "dc", "pgp", "kjv"]

CORPUS_PATHS = {
    "bom": "data/bom.txt",
    "kjv": "data/kjv.txt",
    "dc": "data/dc.txt",
    "pgp": "data/pgp.txt",
}

CORPUS_COLORS = {
    "bom": "#1f6f8b",
    "ot": "#8c5a2b",
    "nt": "#9b1b30",
    "dc": "#5b4b8a",
    "pgp": "#2f7d5b",
    "kjv": "#8c5a2b",
}

CSV_COLUMNS = [
    "pair", "cosine_similarity",
    "query_corpus", "query_book", "query_chapter", "query_verse", "query_end_verse", "query_ref",
    "source_corpus", "source_book", "source_chapter", "source_verse", "source_end_verse", "source_ref",
    "relationship", "secondary_relationship", "confidence", "needs_review", "review_reasons",
    "reproduced_share", "extent_reading",
    "classifier_query_span", "classifier_source_span",
    "guard_applied", "explanation", "evidence", "model", "rubric_version", "error",
    "query_testament", "source_testament",
]

# The KJV run is one classification pass, but the two testaments are different
# neighbourhoods, so the visualisation and the CSV both name them.
OT_BOOKS = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
    "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
    "Malachi",
}
NT_BOOKS = {
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians",
    "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
}


def testament_of(corpus, book):
    """'OT' or 'NT' for a Bible book, '' for anything else."""
    if corpus != "kjv" or not book:
        return ""
    if book in OT_BOOKS:
        return "OT"
    if book in NT_BOOKS:
        return "NT"
    return ""


def viz_corpus(corpus, book):
    """Split the single kjv run into two painted groups for the visualisation."""
    t = testament_of(corpus, book)
    return {"OT": "ot", "NT": "nt"}.get(t, corpus)


def parse_ref(ref):
    """'1 Nephi 21:22' -> ('1 Nephi', 21); 'D&C 76:22' -> ('D&C', 76)."""
    try:
        book, rest = str(ref).rsplit(" ", 1)
        chapter = int(rest.split(":")[0])
    except (ValueError, AttributeError):
        return None, None
    return book.strip(), chapter


def parse_span(span):
    """'22-24' -> (22, 24); '22' -> (22, 22); anything odd -> (None, None)."""
    text = str(span).strip()
    if not text or text.lower() == "nan":
        return None, None
    start, _, end = text.partition("-")
    try:
        low = int(start)
        high = int(end) if end else low
    except ValueError:
        return None, None
    return low, high


def load_rows(journal_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(journal_dir, "*.jsonl"))):
        pair = Path(path).stem
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            row = dict(record["row"])
            row["pair"] = pair
            rows.append(row)
    return rows


def to_frame(rows):
    frame = pd.DataFrame(rows)
    for side in ("query", "source"):
        books, chapters, starts, ends = [], [], [], []
        for ref, span in zip(frame[f"{side}_ref"], frame[f"{side}_span"]):
            book, chapter = parse_ref(ref)
            low, high = parse_span(span)
            books.append(book)
            chapters.append(chapter)
            starts.append(low)
            ends.append(high)
        frame[f"{side}_book"] = books
        frame[f"{side}_chapter"] = chapters
        frame[f"{side}_verse"] = starts
        frame[f"{side}_end_verse"] = ends
        frame[f"{side}_testament"] = [
            testament_of(corpus, book)
            for corpus, book in zip(frame[f"{side}_corpus"], books)
        ]

    frame["needs_review"] = frame["needs_review"].fillna(False).astype(bool)
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["reproduced_share"] = pd.to_numeric(frame["reproduced_share"], errors="coerce")
    return frame


def build_csv(frame, path):
    ordered = frame.reindex(columns=CSV_COLUMNS).sort_values(
        ["pair", "cosine_similarity"], ascending=[True, False]
    )
    ordered.to_csv(path, index=False)
    return ordered


def node_id(corpus, book):
    return f"{corpus}:{book}"


def load_verse_texts(paths):
    """(corpus, book, chapter, verse) -> verse text, for every corpus present."""
    texts = {}
    for corpus_id, path in paths.items():
        if not Path(path).exists():
            continue
        for verse in load_corpus(corpus_id, path):
            texts[(verse["corpus"], verse["book"], verse["chapter"], verse["verse"])] = verse["text"]
    return texts


def span_key(corpus, book, chapter, start, end):
    return f"{corpus}|{book}|{chapter}|{start}-{end}"


def build_span_texts(frame, texts):
    """The unique verse spans, keyed so pairs can point at them.

    Storing each passage once keeps the payload small: 3,000 pairs reference far
    fewer distinct spans.
    """
    spans = {}

    def key_for(corpus, book, chapter, start, end):
        if book is None or chapter is None or start is None:
            return None
        key = span_key(corpus, book, chapter, start, end)
        if key in spans:
            return key
        parts = []
        for verse in range(int(start), int(end) + 1):
            text = texts.get((corpus, book, chapter, verse))
            if text:
                parts.append(f"{chapter}:{verse} {text}")
        spans[key] = " ".join(parts)
        return key

    for row in frame.itertuples(index=False):
        key_for(row.query_corpus, row.query_book, row.query_chapter,
                row.query_verse, row.query_end_verse)
        key_for(row.source_corpus, row.source_book, row.source_chapter,
                row.source_verse, row.source_end_verse)

    return spans


def build_viz_data(frame, path, texts=None):
    if texts is None:
        texts = load_verse_texts(CORPUS_PATHS)

    classified = frame[frame["error"].fillna("").astype(str).str.strip() == ""]
    classified = classified.dropna(subset=["query_book", "source_book"])
    spans = build_span_texts(classified, texts)
    nodes = {}
    edges = defaultdict(lambda: {"count": 0, "labels": Counter(), "cosines": []})
    pairs = []

    for row in classified.itertuples(index=False):
        qc = viz_corpus(row.query_corpus, row.query_book)
        sc = viz_corpus(row.source_corpus, row.source_book)
        query = node_id(qc, row.query_book)
        source = node_id(sc, row.source_book)
        key = (query, source) if query <= source else (source, query)

        if query == source:
            sides = [(query, qc, row.query_book)]
        else:
            sides = [(query, qc, row.query_book),
                     (source, sc, row.source_book)]

        for node, corpus, book in sides:
            entry = nodes.setdefault(node, {
                "id": node, "corpus": corpus, "book": book,
                "pairs": 0, "labels": Counter(), "max_cosine": 0.0,
            })
            entry["pairs"] += 1
            entry["labels"][row.relationship] += 1
            entry["max_cosine"] = max(entry["max_cosine"], float(row.cosine_similarity))

        edge = edges[key]
        edge["count"] += 1
        edge["labels"][row.relationship] += 1
        edge["cosines"].append(float(row.cosine_similarity))

        pairs.append({
            "id": len(pairs),
            "q": query,
            "s": source,
            "qc": row.query_ref,
            "sc": row.source_ref,
            "qk": span_key(row.query_corpus, row.query_book, row.query_chapter,
                           row.query_verse, row.query_end_verse),
            "sk": span_key(row.source_corpus, row.source_book, row.source_chapter,
                           row.source_verse, row.source_end_verse),
            "cos": round(float(row.cosine_similarity), 4),
            "label": row.relationship,
            "conf": None if pd.isna(row.confidence) else round(float(row.confidence), 3),
            "rep": None if pd.isna(row.reproduced_share) else round(float(row.reproduced_share), 3),
            "review": bool(row.needs_review),
            "expl": str(row.explanation or ""),
            "ev": str(row.evidence or ""),
        })

    node_list = []
    for entry in nodes.values():
        node_list.append({
            "id": entry["id"],
            "corpus": entry["corpus"],
            "book": entry["book"],
            "pairs": entry["pairs"],
            "labels": dict(entry["labels"]),
            "max_cosine": round(entry["max_cosine"], 4),
        })

    edge_list = []
    for (a, b), entry in edges.items():
        cosines = entry["cosines"]
        edge_list.append({
            "source": a,
            "target": b,
            "count": entry["count"],
            "labels": dict(entry["labels"]),
            "mean_cosine": round(sum(cosines) / len(cosines), 4),
            "max_cosine": round(max(cosines), 4),
        })

    used_corpora = {n["corpus"] for n in node_list}

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpora": {
            corpus: {"label": CORPUS_LABELS[corpus], "color": CORPUS_COLORS[corpus]}
            for corpus in CORPUS_ORDER if corpus in used_corpora
        },
        "totals": {
            "pairs": len(pairs),
            "nodes": len(node_list),
            "edges": len(edge_list),
            "spans": len(spans),
        },
        "nodes": node_list,
        "edges": edge_list,
        "spans": spans,
        "pairs": pairs,
    }
    Path(path).write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    # A JS wrapper so the page also works when opened straight from disk
    # (file:// blocks fetch of a sibling JSON file).
    js_path = Path(path).with_suffix(".js")
    js_path.write_text(
        "window.VIZ_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journals", default="classifications")
    parser.add_argument("--csv", default="matrix_classifications.csv")
    parser.add_argument("--viz-data", default="viz/data.json")
    args = parser.parse_args(argv)

    rows = load_rows(args.journals)
    frame = to_frame(rows)

    csv_frame = build_csv(frame, args.csv)
    Path(args.viz_data).parent.mkdir(parents=True, exist_ok=True)
    payload = build_viz_data(frame, args.viz_data)

    # the viz folder is deployed on its own, so it carries its own copy
    download = Path(args.viz_data).parent / "matrix_classifications.csv"
    download.write_bytes(Path(args.csv).read_bytes())

    print(f"{len(csv_frame)} rows -> {args.csv} ({os.path.getsize(args.csv)/1e6:.2f} MB)")
    print(f"{payload['totals']['nodes']} books, {payload['totals']['edges']} book pairs, "
          f"{payload['totals']['pairs']} classified pairs -> {args.viz_data} "
          f"({os.path.getsize(args.viz_data)/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
