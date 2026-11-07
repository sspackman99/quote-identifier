"""Classify every pair CSV the matrix produced, in one overnight pass.

`run_matrix.py` writes one pair CSV per query/source pair. This walks those files
in a fixed order, applies a per-pair similarity cutoff, and calls the classifier
through `classify_pairs`, which already handles collapsing, orientation, the repair
retry and the resumable journal.

Cutoffs are per pair TYPE, because the two distributions differ: a within-corpus
run is full of near-duplicate passages (parallel accounts, the same chapter quoted
twice), so it needs a higher bar before a model call is worth spending, while a
cross-corpus run's distribution sits lower.

    python classify_matrix.py                     # every pair present in pairs/
    python classify_matrix.py --pairs bom-kjv --min-similarity 0.90
"""

import argparse
import json
import sys
import time
from pathlib import Path

import classify_pairs

DEFAULT_WITHIN_CUTOFF = 0.85
DEFAULT_CROSS_CUTOFF = 0.80

CORPORA = {
    "bom": "data/bom.txt",
    "kjv": "data/kjv.txt",
    "dc": "data/dc.txt",
    "pgp": "data/pgp.txt",
}

# The order to work through the matrix: cross-corpus pairs first (they are where
# the interesting relationships are), then the within-corpus runs, with the huge
# Bible-to-itself run last because it is mostly synoptic parallels.
PRIORITY = [
    ("bom", "kjv"), ("dc", "kjv"), ("pgp", "kjv"),
    ("dc", "bom"), ("pgp", "bom"), ("dc", "pgp"),
    ("bom", "bom"), ("dc", "dc"), ("pgp", "pgp"), ("kjv", "kjv"),
]


def pair_name(query, source):
    return f"{query}-{source}"


def parse_selection(values):
    if not values:
        return list(PRIORITY)
    selected = []
    for value in values:
        query, _, source = value.partition("-")
        if not source:
            raise SystemExit(f"--pairs needs QUERY-SOURCE, got {value!r}")
        selected.append((query, source))
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="pairs")
    parser.add_argument("--output-dir", default="classifications")
    parser.add_argument("--pairs", action="append", metavar="QUERY-SOURCE",
                        help="only these pairs (repeatable)")
    parser.add_argument("--min-similarity", type=float, default=None,
                        help="override the per-type cutoff for every pair")
    parser.add_argument("--within-cutoff", type=float, default=DEFAULT_WITHIN_CUTOFF)
    parser.add_argument("--cross-cutoff", type=float, default=DEFAULT_CROSS_CUTOFF)
    parser.add_argument("--pair-cutoff", action="append", metavar="QUERY-SOURCE=VALUE",
                        help="override the cutoff for one pair (repeatable), e.g. "
                             "kjv-kjv=0.90")
    parser.add_argument("--limit", type=int, default=0,
                        help="max pairs to classify per pair file (0 = all)")
    parser.add_argument("--budget", type=int, default=0,
                        help="stop starting new pairs after this many journal records")
    parser.add_argument("--corpus-dir", default="data")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = parse_selection(args.pairs)

    overrides = {}
    for value in args.pair_cutoff or []:
        name, separator, cutoff = value.partition("=")
        if not separator:
            raise SystemExit(f"--pair-cutoff needs QUERY-SOURCE=VALUE, got {value!r}")
        overrides[name.strip()] = float(cutoff)

    ran = []

    for query_id, source_id in selections:
        name = pair_name(query_id, source_id)
        pair_csv = input_dir / f"{name}.csv"
        if not pair_csv.exists():
            print(f"skip {name}: no {pair_csv}")
            continue

        if query_id == source_id:
            cutoff = args.within_cutoff
            orientation = ["--corpus", str(Path(args.corpus_dir) / f"{query_id}.txt"),
                           "--corpus-id", query_id]
        else:
            cutoff = args.cross_cutoff
            orientation = ["--corpus", str(Path(args.corpus_dir) / f"{query_id}.txt"),
                           "--corpus-id", query_id,
                           "--source-corpus", str(Path(args.corpus_dir) / f"{source_id}.txt"),
                           "--source-corpus-id", source_id]
        if args.min_similarity is not None:
            cutoff = args.min_similarity
        cutoff = overrides.get(name, cutoff)

        journal = output_dir / f"{name}.jsonl"
        output = output_dir / f"{name}.csv"

        if args.budget:
            already = 0
            for previous in output_dir.glob("*.jsonl"):
                with open(previous, encoding="utf-8") as handle:
                    already += sum(1 for line in handle if line.strip())
            if already >= args.budget:
                print(f"budget reached ({already} records); stopping before {name}")
                break

        print(f"\n=== {name} (cosine >= {cutoff}) ===", flush=True)
        started = time.time()
        status = classify_pairs.main([
            "--input", str(pair_csv),
            "--output", str(output),
            "--journal", str(journal),
            "--min-similarity", str(cutoff),
            "--limit", str(args.limit),
            *orientation,
        ])
        ran.append({"pair": name, "cutoff": cutoff, "status": status,
                    "seconds": round(time.time() - started, 1),
                    "output": str(output)})

    summary = output_dir / "matrix_classification.json"
    summary.write_text(json.dumps({
        "pairs": ran,
        "cutoffs": {"within": args.within_cutoff, "cross": args.cross_cutoff},
        "override": args.min_similarity,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
