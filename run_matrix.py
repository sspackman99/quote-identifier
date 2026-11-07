"""Run the whole within/cross corpus matrix from one embedding pass.

Ten runs over four corpora: four within-corpus runs and six directed cross runs.
Preprocessing, common-phrase filtering and embedding happen ONCE; each pair then
gets its own CSV and metadata sidecar, plus a manifest recording what was
produced, so a night's work is auditable in one file.

The pair list is directed: "dc-bom" means D&C windows are the query and Book of
Mormon windows are the source, which is also the orientation the classifier uses.

Usage:
    QI_EMBEDDINGS_URL=http://host:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python run_matrix.py --output-dir pairs --cache .embed_cache/embeddings.sqlite3
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from embeddings_cache import EmbeddingCache
from main import build_state, embedding_model_name, pairs_for, write_pairs

DEFAULT_CORPORA = {
    "bom": "data/bom.txt",
    "kjv": "data/kjv.txt",
    "dc": "data/dc.txt",
    "pgp": "data/pgp.txt",
}

# Four within-corpus runs, then six cross runs, each in the direction that makes
# the older text the source. dc-pgp is genuinely ambiguous (both are 1830s
# restorations), so it is run one way here and can be run the other way by adding
# "pgp:dc" to --pair.
MATRIX = (
    ("bom", "bom"), ("kjv", "kjv"), ("dc", "dc"), ("pgp", "pgp"),
    ("bom", "kjv"), ("dc", "kjv"), ("pgp", "kjv"),
    ("dc", "bom"), ("pgp", "bom"), ("dc", "pgp"),
)

TOP_K = 5


def parse_corpora(values):
    corpora = dict(DEFAULT_CORPORA)
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--corpora needs ID=PATH, got {value!r}")
        corpus_id, _, path = value.partition("=")
        corpora[corpus_id.strip()] = path.strip()
    return corpora


def parse_pairs(values):
    if not values:
        return list(MATRIX)
    pairs = []
    for value in values:
        query, sep, source = value.partition(":")
        if not sep or not query or not source:
            raise SystemExit(f"--pair needs QUERY:SOURCE, got {value!r}")
        pairs.append((query.strip(), source.strip()))
    return pairs


def summarise(frame):
    if frame.empty:
        return {"rows": 0}
    scores = frame["cosine_similarity"]
    same_chapter = (frame["chapter_1"] == frame["chapter_2"]) & (
        frame["book_1"] == frame["book_2"]
    )
    return {
        "rows": int(len(frame)),
        "max": round(float(scores.max()), 4),
        "median": round(float(scores.median()), 4),
        "p90": round(float(frame["cosine_similarity"].quantile(0.90)), 4),
        "rows_above_0.80": int((scores >= 0.80).sum()),
        "rows_above_0.70": int((scores >= 0.70).sum()),
        "same_chapter_rows": int(same_chapter.sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", action="append", metavar="ID=PATH",
                        help="override a corpus path (repeatable)")
    parser.add_argument("--pair", action="append", metavar="QUERY:SOURCE",
                        help="run only these pairs (repeatable); default is the full matrix")
    parser.add_argument("--output-dir", default="pairs")
    parser.add_argument("--cache", default=".embed_cache/embeddings.sqlite3",
                        help="SQLite embedding cache (default: %(default)s)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--filter-metric", default="document",
                        choices=["document", "occurrence"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    corpora_by_id = parse_corpora(args.corpora)
    pairs = parse_pairs(args.pair)

    unknown = {cid for pair in pairs for cid in pair if cid not in corpora_by_id}
    if unknown:
        parser.exit(2, f"{parser.prog}: error: unknown corpora {sorted(unknown)}; "
                       f"known: {sorted(corpora_by_id)}\n")

    needed = sorted({cid for pair in pairs for cid in pair})
    corpora = [(cid, corpora_by_id[cid]) for cid in needed]
    for corpus_id, path in corpora:
        if not Path(path).exists():
            parser.exit(2, f"{parser.prog}: error: no corpus file for {corpus_id}: {path}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = None
    if not args.no_cache:
        cache = EmbeddingCache(args.cache, embedding_model_name())

    started = time.time()
    print(f"Building state for {', '.join(needed)}")
    state = build_state(corpora, filter_metric=args.filter_metric, cache=cache)
    if cache is not None:
        print(f"Embeddings: {cache.hits} from cache, {cache.misses} computed")
    state_seconds = time.time() - started

    results = []
    for query_id, source_id in pairs:
        name = f"{query_id}-{source_id}"
        pair_started = time.time()
        print(f"\n=== {name} ===")
        found, directed = pairs_for(state, query_id, source_id, args.top_k)
        frame = write_pairs(
            state, found, output_dir / f"{name}.csv",
            query_id=query_id, source_id=source_id,
            top_k=args.top_k, directed_candidates=directed,
        )
        entry = {
            "pair": name,
            "query": query_id,
            "source": source_id,
            "directed_candidates": directed,
            "seconds": round(time.time() - pair_started, 1),
        }
        entry.update(summarise(frame))
        results.append(entry)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_model": embedding_model_name(),
        "embedding_endpoint": os.environ.get("QI_EMBEDDINGS_URL", "local"),
        "filter_metric": args.filter_metric,
        "top_k": args.top_k,
        "corpora": state["corpora"],
        "state_seconds": round(state_seconds, 1),
        "pairs": results,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("\npair                rows     max   median   >0.80")
    for entry in results:
        print(f"  {entry['pair']:<16} {entry['rows']:>7}  "
              f"{entry.get('max', 0):>5.3f}  {entry.get('median', 0):>6.3f}  "
              f"{entry.get('rows_above_0.80', 0):>7}")
    print(f"\nmanifest: {manifest_path}")
    print(f"state built in {state_seconds:.1f}s")

    if cache is not None:
        cache.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
