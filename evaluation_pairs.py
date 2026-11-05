"""Labelled cross-corpus pairs for retrieval evaluation.

Two questions are kept apart on purpose:

  retrieval       does the correct source chapter appear near the top?
  identification  is the relationship actually quotation, paraphrase, allusion,
                  or coincidence?

These labels only support the first. Every positive names a KJV chapter the Book
of Mormon chapter is textually related to; the `confidence` field records how
strong that claim is and `note` says why, so a reviewer can check the labels
without ever seeing retrieval scores.

Kinds:
  quotation          near-verbatim reproduction of the KJV chapter
  partial_quotation  reproduces part of the chapter (a few verses), not all of it
  paraphrase         follows the source closely with rewording
  allusion           genuine textual dependence but loose: theme and phrasing echo
                     the source without reproducing it; low confidence by design
  shared_phrases     control: KJV-shaped religious vocabulary, no single source
  unrelated          control: narrative with no KJV source

Split: derived from the query string, not from its position, so it stays stable
when pairs are added or reordered. Roughly one query in three lands in the
held-out "test" split; thresholds are chosen on "dev" and reported on "test".
"""

import hashlib

QUOTATION_PAIRS = [
    # 1 Nephi 20-21 reproduce Isaiah 48-49
    {"query": "1 Nephi 20", "expected": "Isaiah 48", "confidence": "high"},
    {"query": "1 Nephi 21", "expected": "Isaiah 49", "confidence": "high"},
    # 2 Nephi 7-8 reproduce Isaiah 50-51
    {"query": "2 Nephi 7", "expected": "Isaiah 50", "confidence": "high"},
    {"query": "2 Nephi 8", "expected": "Isaiah 51", "confidence": "high"},
    # 2 Nephi 12-24 reproduce Isaiah 2-14, chapter for chapter
    {"query": "2 Nephi 12", "expected": "Isaiah 2", "confidence": "high"},
    {"query": "2 Nephi 13", "expected": "Isaiah 3", "confidence": "high"},
    {"query": "2 Nephi 14", "expected": "Isaiah 4", "confidence": "high"},
    {"query": "2 Nephi 15", "expected": "Isaiah 5", "confidence": "high"},
    {"query": "2 Nephi 16", "expected": "Isaiah 6", "confidence": "high"},
    {"query": "2 Nephi 17", "expected": "Isaiah 7", "confidence": "high"},
    {"query": "2 Nephi 18", "expected": "Isaiah 8", "confidence": "high"},
    {"query": "2 Nephi 19", "expected": "Isaiah 9", "confidence": "high"},
    {"query": "2 Nephi 20", "expected": "Isaiah 10", "confidence": "high"},
    {"query": "2 Nephi 21", "expected": "Isaiah 11", "confidence": "high"},
    {"query": "2 Nephi 22", "expected": "Isaiah 12", "confidence": "high"},
    {"query": "2 Nephi 23", "expected": "Isaiah 13", "confidence": "high"},
    {"query": "2 Nephi 24", "expected": "Isaiah 14", "confidence": "high"},
    # Mosiah 14 reproduces Isaiah 53
    {"query": "Mosiah 14", "expected": "Isaiah 53", "confidence": "high"},
    # 3 Nephi 22 reproduces Isaiah 54
    {"query": "3 Nephi 22", "expected": "Isaiah 54", "confidence": "high"},
    # 3 Nephi 24-25 reproduce Malachi 3-4
    {"query": "3 Nephi 24", "expected": "Malachi 3", "confidence": "high"},
    {"query": "3 Nephi 25", "expected": "Malachi 4", "confidence": "high"},
]

PARTIAL_QUOTATION_PAIRS = [
    # Only part of each chapter is taken over; the rest is independent material.
    {"query": "Mosiah 12", "expected": "Isaiah 52", "confidence": "high",
     "note": "vs 20-24 quote Isaiah 52:7-10; remainder is Mosiah's narrative"},
    {"query": "3 Nephi 20", "expected": "Isaiah 52", "confidence": "high",
     "note": "vs 23-26 quote Isaiah 52:8-10; remainder is Jesus speaking"},
]

PARAPHRASE_PAIRS = [
    # 3 Nephi 12-14 follow the Sermon on the Mount closely but with edits
    {"query": "3 Nephi 12", "expected": "Matthew 5", "confidence": "high"},
    {"query": "3 Nephi 13", "expected": "Matthew 6", "confidence": "high"},
    {"query": "3 Nephi 14", "expected": "Matthew 7", "confidence": "high"},
    # Moroni 7:45-48 track 1 Corinthians 13:4-8 on charity, lightly reworded
    {"query": "Moroni 7", "expected": "1 Corinthians 13", "confidence": "medium",
     "note": "vs 45-48 track 1 Cor 13:4-8; rest of the chapter is independent"},
]

ALLUSION_PAIRS = [
    # Thematic dependence without reproduced wording. Low confidence by design:
    # these are judgements about the text, not citations.
    {"query": "Alma 7", "expected": "Isaiah 53", "confidence": "low",
     "note": "vs 10-12 take up the servant's pains, not his wording"},
    {"query": "Mosiah 15", "expected": "Isaiah 53", "confidence": "low",
     "note": "suffering-servant themes, no shared phrasing"},
    {"query": "3 Nephi 19", "expected": "John 17", "confidence": "low",
     "note": "intercessory prayer echoing John 17 in structure, not wording"},
]

# Controls: chapters with no single KJV source. Two flavours, because they test
# different failure modes.
SHARED_PHRASE_CONTROLS = [
    # KJV-shaped religious vocabulary that quotes no particular chapter
    {"query": "Moroni 10"},
    {"query": "2 Nephi 2"},
    {"query": "Alma 42"},
    {"query": "Mosiah 4"},
]

UNRELATED_CONTROLS = [
    {"query": "Alma 43"},     # war narrative
    {"query": "Alma 50"},     # war and fortifications
    {"query": "Ether 1"},     # genealogy
    {"query": "Ether 6"},     # voyage narrative
    {"query": "Moroni 9"},    # a captain's epistle
    {"query": "1 Nephi 16"},  # travel narrative
    {"query": "3 Nephi 4"},   # battle
    {"query": "Helaman 11"},  # famine and preaching
]


def split_for(query: str) -> str:
    """Deterministic dev/test assignment derived from the query string itself."""
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return "test" if int(digest, 16) % 3 == 0 else "dev"


def all_pairs():
    """Every labelled pair, with kind and split; controls carry expected=None."""
    rows = []
    for kind, source in (
        ("quotation", QUOTATION_PAIRS),
        ("partial_quotation", PARTIAL_QUOTATION_PAIRS),
        ("paraphrase", PARAPHRASE_PAIRS),
        ("allusion", ALLUSION_PAIRS),
    ):
        for row in source:
            rows.append({**row, "kind": kind})

    for kind, source in (
        ("shared_phrases", SHARED_PHRASE_CONTROLS),
        ("unrelated", UNRELATED_CONTROLS),
    ):
        for row in source:
            rows.append({"query": row["query"], "expected": None,
                         "kind": kind, "confidence": "high"})

    for row in rows:
        row.setdefault("note", "")
        row["split"] = split_for(row["query"])
    return rows


def positives(rows=None):
    rows = rows if rows is not None else all_pairs()
    return [row for row in rows if row["expected"]]


def controls(rows=None):
    rows = rows if rows is not None else all_pairs()
    return [row for row in rows if not row["expected"]]
