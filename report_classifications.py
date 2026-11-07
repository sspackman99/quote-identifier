"""Render completed classifications as a browsable HTML page.

Reads the journal written by classify_pairs.py (so it works on a run in progress,
without touching the CSV), pulls the verse text for each cited span, and writes
one self-contained HTML file. Also prints a short digest for the terminal.

Usage:
    python report_classifications.py                    # -> classifications_report.html
    python report_classifications.py --journal x.jsonl --output y.html
"""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from preprocess import preprocess_corpus

LABELS = (
    "quotation", "partial_quotation", "paraphrase", "allusion",
    "interpretation", "shared_vocabulary", "unrelated",
)

CHIP = {
    "quotation": ("#1b5e20", "#e6f4ea"),
    "partial_quotation": ("#2e7d32", "#f1f8e9"),
    "paraphrase": ("#e65100", "#fff3e0"),
    "allusion": ("#6a1b9a", "#f3e5f5"),
    "interpretation": ("#00695c", "#e0f2f1"),
    "shared_vocabulary": ("#546e7a", "#eceff1"),
    "unrelated": ("#b71c1c", "#ffebee"),
    "ERROR": ("#b71c1c", "#ffebee"),
}


def read_records(journal=None, journal_dir=None):
    """Read one journal, or every journal in a directory, tagging the pair.

    The pair name comes from the file name (bom-kjv.jsonl -> "bom-kjv"), which is
    what makes a combined matrix report possible.
    """
    paths = []
    if journal:
        paths.append((None, Path(journal)))
    if journal_dir:
        paths += [(path.stem, path) for path in sorted(Path(journal_dir).glob("*.jsonl"))]

    records = []
    for pair, path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["pair"] = pair
            records.append(record)

    # last record wins for a repeated key (a re-run may rewrite one)
    by_key = {(record.get("pair"), record["key"]): record for record in records}
    return sorted(by_key.values(),
                  key=lambda record: -record["row"]["cosine_similarity"])


def load_texts(*specs):
    """(corpus, book, chapter, verse) -> verse text, for each corpus supplied."""
    texts = {}
    for corpus_id, path in specs:
        for verse in preprocess_corpus(corpus_id, str(path)):
            texts[(verse["corpus"], verse["book"], verse["chapter"], verse["verse"])] = verse["text"]
    return texts


def span_text(texts, corpus, ref, span):
    """'2 Nephi 6:6' + '6-6' -> the verse text(s) of that span."""
    try:
        book, rest = ref.rsplit(" ", 1)
        chapter = int(rest.split(":")[0])
        start, _, end = str(span).partition("-")
        low, high = int(start), int(end or start)
    except (ValueError, AttributeError):
        return ""

    lines = []
    for verse in range(low, high + 1):
        text = texts.get((corpus, book, chapter, verse))
        if text:
            lines.append(f"<span class='v'>{chapter}:{verse}</span> {html.escape(text)}")
    return "<br>".join(lines)


def chip(label):
    color, background = CHIP.get(label, ("#37474f", "#eceff1"))
    return (f"<span class='chip' style='color:{color};background:{background}'>"
            f"{html.escape(label)}</span>")


def render(records, args):
    done = [record for record in records if not record["row"].get("error")]
    failed = [record for record in records if record["row"].get("error")]
    labels = Counter(record["row"]["relationship"] for record in done)
    confidences = [record["row"]["confidence"] for record in done
                   if record["row"].get("confidence") is not None]
    median = sorted(confidences)[len(confidences) // 2] if confidences else 0
    reviewed = sum(1 for record in done if record["row"].get("needs_review") is True)
    retried = sum(1 for record in records if int(record["row"].get("attempts") or 1) > 1)

    texts = load_texts(*args.corpus_specs)

    cross = any(
        (record["row"].get("query_corpus") or args.corpus_id)
        != (record["row"].get("source_corpus") or args.corpus_id)
        for record in done
    )
    title = (f"{args.corpus_id.upper()} &rarr; {args.source_corpus_id.upper()} cross-corpus pairs"
             if cross else "Pair classifications &mdash; " + args.corpus_id.upper())

    cards = [
        ("classified", f"{len(done)}"),
        ("malformed / failed", f"{len(failed)}"),
        ("retried after a bad answer", f"{retried}"),
        ("median confidence", f"{median:.2f}"),
        ("flagged for review", f"{reviewed}"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='num'>{value}</div><div class='cap'>{html.escape(name)}</div></div>"
        for name, value in cards
    )

    counts_html = "".join(
        f"<div class='count'>{chip(label)}<b>{labels.get(label, 0)}</b></div>"
        for label in LABELS
    )

    pair_names = sorted({record["pair"] for record in records if record.get("pair")})
    matrix = len(pair_names) > 1
    if matrix:
        title = f"Corpus matrix &mdash; {len(pair_names)} pairs, {len(done)} classifications"
        summary_rows = []
        for name in pair_names:
            subset = [record["row"] for record in done if record.get("pair") == name]
            counts = Counter(row["relationship"] for row in subset)
            summary_rows.append(
                "<tr><td class='pairname'>" + html.escape(name) + "</td>"
                + f"<td>{len(subset)}</td>"
                + "".join(f"<td>{counts.get(label, 0)}</td>" for label in LABELS)
                + f"<td>{sum(1 for row in subset if row.get('needs_review') is True)}</td></tr>"
            )
        summary = (
            "<h2>Per pair</h2><table class='summary'><thead><tr><th>pair</th><th>n</th>"
            + "".join(f"<th>{label[:11]}</th>" for label in LABELS)
            + "<th>review</th></tr></thead><tbody>"
            + "".join(summary_rows)
            + "</tbody></table>"
        )
    else:
        summary = ""

    rows = []
    for record in done:
        row = record["row"]
        query_corpus = row.get("query_corpus") or args.corpus_id
        source_corpus = row.get("source_corpus") or args.corpus_id
        query_label = f"A &middot; query &middot; {query_corpus}" if cross else "A &middot; query"
        source_label = f"B &middot; source &middot; {source_corpus}" if cross else "B &middot; source"
        rows.append(f"""
        <tr>
          {f"<td class='pairname'>{html.escape(str(record.get('pair')))}</td>" if matrix else ""}
          <td class='cos'>{row['cosine_similarity']:.3f}</td>
          <td class='passage'>
            <div class='ref'>{query_label} &middot; {html.escape(str(row['query_ref']))}</div>
            <div class='verse'>{span_text(texts, query_corpus, row['query_ref'], row['query_span'])}</div>
          </td>
          <td class='passage'>
            <div class='ref'>{source_label} &middot; {html.escape(str(row['source_ref']))}</div>
            <div class='verse'>{span_text(texts, source_corpus, row['source_ref'], row['source_span'])}</div>
          </td>
          <td class='verdict'>
            {chip(row['relationship'])}
            {chip(row['secondary_relationship']) if row.get('secondary_relationship') else ''}
            <div class='conf'>confidence {row['confidence']:.2f}</div>
            <div class='conf'>reproduced {('%.0f%%' % (row['reproduced_share']*100)) if row.get('reproduced_share') is not None else '-'}</div>
            {f"<div class='review'>needs review: {html.escape(str(row['review_reasons']))}</div>" if row.get('needs_review') else ''}
            {f"<div class='review'>guard: {html.escape(str(row['guard_applied']))}</div>" if row.get('guard_applied') else ''}
          </td>
          <td class='why'>
            <div class='expl'>{html.escape(str(row['explanation']))}</div>
            {f"<div class='ev'>&ldquo;{html.escape(str(row['evidence']))}&rdquo;</div>" if row.get('evidence') else ''}
          </td>
        </tr>""")

    error_rows = "".join(
        f"<tr><td>{html.escape(str(record['row']['query_ref']))} &rarr; "
        f"{html.escape(str(record['row']['source_ref']))}</td>"
        f"<td>{html.escape(str(record['row'].get('error')))}</td></tr>"
        for record in failed
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Pair classifications — Book of Mormon</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:28px 20px 80px; font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         color:#1a1a1a; background:#fafafa; }}
  header, main {{ max-width:1180px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .sub {{ color:#666; margin-bottom:20px; }}
  .cards {{ display:flex; gap:10px; flex-wrap:wrap; margin:18px 0 10px; }}
  .card {{ background:#fff; border:1px solid #e3e3e3; border-radius:10px; padding:10px 14px; min-width:130px; }}
  .card .num {{ font-size:22px; font-weight:650; }}
  .card .cap {{ font-size:12px; color:#666; }}
  .counts {{ display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 22px; font-size:13px; }}
  .count b {{ margin-left:6px; }}
  .chip {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600;
           letter-spacing:.01em; white-space:nowrap; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid #e3e3e3; border-radius:10px; overflow:hidden; }}
  thead th {{ text-align:left; font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:#777;
              background:#f4f4f4; padding:9px 12px; border-bottom:1px solid #e3e3e3; position:sticky; top:0; }}
  td {{ padding:11px 12px; border-bottom:1px solid #eee; vertical-align:top; }}
  tr:last-child td {{ border-bottom:none; }}
  td.cos {{ font-variant-numeric:tabular-nums; font-weight:600; width:56px; }}
  td.passage {{ width:27%; }}
  .ref {{ font-size:12px; font-weight:650; color:#333; margin-bottom:3px; }}
  .verse {{ font-size:13px; color:#444; }}
  .v {{ color:#999; font-size:11px; margin-right:3px; }}
  td.verdict {{ width:190px; }}
  .conf {{ font-size:12px; color:#555; margin-top:4px; }}
  .review {{ font-size:11.5px; color:#b26a00; margin-top:4px; }}
  td.why {{ font-size:13px; }}
  .expl {{ margin-bottom:6px; }}
  .ev {{ color:#4a4a4a; font-style:italic; font-size:12.5px; border-left:3px solid #ddd; padding-left:9px; }}
  h2 {{ font-size:15px; margin:30px 0 8px; }}
  .errors table {{ font-size:13px; }}
  td.pairname {{ font-size:12px; font-weight:650; color:#444; white-space:nowrap; }}
  table.summary {{ margin-bottom:26px; font-size:13px; }}
  table.summary td, table.summary th {{ padding:6px 10px; }}
  footer {{ max-width:1180px; margin:26px auto 0; color:#888; font-size:12px; }}
</style></head>
<body>
<header>
  <h1>{title}</h1>
  <div class="sub">cosine similarity &ge; {args.min_similarity} &middot; rubric {html.escape(args.rubric)}
    &middot; model {html.escape(args.model)} &middot; generated {html.escape(args.generated)}</div>
  <div class="cards">{card_html}</div>
  <div class="counts">{counts_html}</div>
</header>
<main>
  {summary}
  <table>
    <thead><tr>{'<th>pair</th>' if matrix else ''}<th>cos</th><th>A &mdash; the text that may be quoting</th><th>B &mdash; the source</th>
    <th>verdict</th><th>why</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {f"<h2>Failed pairs ({len(failed)})</h2><div class='errors'><table><tbody>{error_rows}</tbody></table></div>" if failed else ""}
</main>
<footer>Orientation: within a corpus the canonically earlier side is B (source), the later side A (query); \
across corpora the named source corpus is B. Rows are distinct verse-span pairs; overlapping windows are \
collapsed before classification.</footer>
</body></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--journal-dir", default=None,
                        help="read every *.jsonl in this directory instead of one journal")
    parser.add_argument("--output", default="classifications_report.html")
    parser.add_argument("--corpora", action="append", metavar="ID=PATH",
                        help="corpus files to load for verse text (repeatable)")
    parser.add_argument("--corpus", default="data/bom.txt")
    parser.add_argument("--corpus-id", default="bom")
    parser.add_argument("--source-corpus", default=None,
                        help="second corpus file, for a cross-corpus classification run")
    parser.add_argument("--source-corpus-id", default=None)
    parser.add_argument("--min-similarity", type=float, default=0.80)
    parser.add_argument("--rubric", default="1.0")
    parser.add_argument("--model", default="Qwen3.5-9B-Q4-Vision")
    parser.add_argument("--generated", default="")
    args = parser.parse_args(argv)

    args.source_corpus_id = args.source_corpus_id or args.corpus_id
    args.corpus_specs = [(args.corpus_id, args.corpus)]
    if args.source_corpus and args.source_corpus_id != args.corpus_id:
        args.corpus_specs.append((args.source_corpus_id, args.source_corpus))
    if args.corpora:
        args.corpus_specs = [tuple(value.split("=", 1)) for value in args.corpora]

    if not args.journal and not args.journal_dir:
        args.journal = "classifications.jsonl"

    records = read_records(args.journal, args.journal_dir)
    Path(args.output).write_text(render(records, args), encoding="utf-8")

    done = [record for record in records if not record["row"].get("error")]
    print(f"{len(records)} journal records ({len(done)} classified) -> {args.output}")
    for label, count in Counter(r["row"]["relationship"] for r in done).most_common():
        print(f"  {label:<20} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
