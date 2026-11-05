"""Span-aware candidate scoring over an embedded window index.

Chapter-level ranking counted how many query windows pointed at a chapter, which
lets a long chapter's incidental matches outvote a short, strong quotation. This
module keeps the evidence instead of only the tally:

  best        best cosine similarity between any query window and the candidate
  support     number of DISTINCT, non-overlapping matched verse ranges in the
              candidate (overlapping windows collapse into one support)
  coverage    share of the candidate chapter's verses touched by those ranges
  query_range / source_range   the verse ranges doing the pointing, as printable
              strings like "36-40" or "1-2,7-10"

Ranking follows the reviewer's rule: best similarity leads, but candidates within
`tie_tolerance` of each other are ordered by support, then coverage — so one
accidental high-similarity window does not automatically beat a candidate backed
by several independent matches.

Annotated spans are used by the EVALUATION only. Nothing here is told where the
quotation is; retrieval has to find it.
"""

def merge_intervals(intervals):
    """Collapse overlapping inclusive (start, end) pairs into disjoint ranges."""
    ordered = sorted((int(start), int(end)) for start, end in intervals)
    merged = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_ranges(intervals):
    """[(1,2),(7,10)] -> '1-2,7-10'; [(5,5)] -> '5'."""
    merged = merge_intervals(intervals)
    parts = []
    for start, end in merged:
        parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def verses_per_chapter(meta):
    """Verses that exist per (book, chapter), derived from the indexed windows."""
    spans = {}
    for window in meta:
        key = (window["book"], window["chapter"])
        spans.setdefault(key, []).append((window["start_verse"], window["end_verse"]))

    return {
        key: sum(end - start + 1 for start, end in merge_intervals(intervals))
        for key, intervals in spans.items()
    }


def collect_matches(meta, index, embeddings, query_indices, neighbors,
                    target_corpus="kjv"):
    """Return every (query window, target window) pair found in the top-k search."""
    import numpy as np

    k = min(neighbors + 1, len(meta))
    matches = []

    for i in query_indices:
        D, I = index.search(np.array([embeddings[i]]), k)
        for score, j in zip(D[0], I[0]):
            j = int(j)
            if j == i:
                continue
            if meta[j]["corpus"] != target_corpus:
                continue
            matches.append({"sim": float(score), "query_index": int(i), "target_index": j})

    return matches


def aggregate_candidates(meta, matches, chapter_sizes=None):
    """Group matches by candidate chapter, keeping the evidence that produced them."""
    if chapter_sizes is None:
        chapter_sizes = verses_per_chapter(meta)

    candidates = {}

    for match in matches:
        target = meta[match["target_index"]]
        query = meta[match["query_index"]]
        key = (target["book"], target["chapter"])

        candidate = candidates.get(key)
        if candidate is None:
            candidate = candidates[key] = {
                "book": target["book"],
                "chapter": target["chapter"],
                "best": match["sim"],
                "matches": 0,
                "source_intervals": [],
                "query_intervals": [],
                "pairs": [],
            }

        candidate["best"] = max(candidate["best"], match["sim"])
        candidate["matches"] += 1
        candidate["source_intervals"].append((target["start_verse"], target["end_verse"]))
        candidate["query_intervals"].append((query["start_verse"], query["end_verse"]))
        candidate["pairs"].append({
            "sim": match["sim"],
            "query_range": f'{query["start_verse"]}-{query["end_verse"]}',
            "source_range": f'{target["start_verse"]}-{target["end_verse"]}',
        })

    for candidate in candidates.values():
        source = merge_intervals(candidate["source_intervals"])
        query = merge_intervals(candidate["query_intervals"])
        candidate["support"] = len(source)
        candidate["source_intervals"] = source
        candidate["query_intervals"] = query
        candidate["source_range"] = format_ranges(source)
        candidate["query_range"] = format_ranges(query)

        size = chapter_sizes.get((candidate["book"], candidate["chapter"]), 0)
        covered = sum(end - start + 1 for start, end in source)
        # Support outside the chapter's own verses cannot exceed 1.0
        candidate["coverage"] = min(1.0, covered / size) if size else 0.0
        # paired evidence: WHICH query range matched WHICH source range, strongest first
        candidate["pairs"].sort(key=lambda pair: -pair["sim"])

    return candidates


def rank_candidates(candidates, tie_tolerance=0.01):
    """Rank candidate chapters with span-aware evidence, in deterministic BANDS.

    Pairwise "within tolerance" comparisons are not transitive: with tolerance
    0.01, A=0.990 ties B=0.985 and B=0.985 ties C=0.980, but A does not tie C, so
    a comparator-based sort can depend on input order. Instead:

      1. sort by best similarity;
      2. take the highest remaining score as the band ANCHOR and group every
         candidate within `tie_tolerance` of that anchor;
      3. order that band by support, then coverage, then best similarity, then name;
      4. repeat with the remaining candidates.

    Band membership is defined by the anchor, so the ordering is a total order and
    independent of the order candidates arrive in.
    """
    remaining = sorted(
        candidates.values(),
        key=lambda candidate: (-candidate["best"], candidate["book"], candidate["chapter"]),
    )

    ordered = []
    while remaining:
        anchor = remaining[0]["best"]
        band = [c for c in remaining if anchor - c["best"] <= tie_tolerance + 1e-12]
        in_band = {id(c) for c in band}
        remaining = [c for c in remaining if id(c) not in in_band]

        band.sort(key=lambda c: (-c["support"], -c["coverage"], -c["best"],
                                 c["book"], c["chapter"]))
        ordered.extend(band)

    for position, candidate in enumerate(ordered, 1):
        candidate["rank"] = position

    return ordered



def span_hit(candidate, annotated_span):
    """Does the candidate's matched source range touch the annotated span?

    'whole' means the annotation covers the chapter, so any support counts.
    """
    if not annotated_span or annotated_span == "whole":
        return True
    wanted = expand_span(annotated_span)
    return bool({verse for start, end in candidate["source_intervals"]
                 for verse in range(start, end + 1)} & wanted)


def query_span_hit(candidate, annotated_span):
    if not annotated_span or annotated_span == "whole":
        return True
    wanted = expand_span(annotated_span)
    return bool({verse for start, end in candidate["query_intervals"]
                 for verse in range(start, end + 1)} & wanted)


def expand_span(span):
    """'1-2,7-10' -> {1,2,7,8,9,10}"""
    verses = set()
    for part in span.split(","):
        if "-" in part:
            start, end = part.split("-")
            verses.update(range(int(start), int(end) + 1))
        else:
            verses.add(int(part))
    return verses
