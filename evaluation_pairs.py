"""Labelled cross-corpus pairs for retrieval evaluation.

Two questions are kept apart:

  retrieval       does the correct source chapter appear near the top?
  identification  is the relationship quotation, paraphrase, allusion,
                  interpretation, or coincidence?

These labels support the first. Kinds:

  quotation           near-verbatim reproduction of the whole chapter
  partial_quotation   reproduces part of the chapter; the span says which verses
  paraphrase          follows the source closely with rewording
  allusion            textual echo without reproduction
  interpretation      explains or retells a source the discourse already quotes or
                      invokes — contextual dependence, NOT independent retrieval.
                      Scored separately and excluded from retrieval recall, because
                      an exposition of Isaiah 53 does not "retrieve" it.
  shared_vocabulary   control: KJV-shaped vocabulary, no identifiable source chapter
  unrelated           control: narrative with no KJV source

Every pair carries query_span/expected_span ("whole" or "a-b"), a note where the
label rests on a judgement, and reviewed=True only when Sam has checked the label
against the text. Unreviewed labels stay flagged so a result is never presented as
if it rested on checked labels.

The dev/test split is assigned per SOURCE FAMILY, not per query, so neighbouring
chapters of one quotation block (2 Nephi 12-24 / Isaiah 2-14) cannot straddle the
split and turn the held-out set into a rerun of development.
"""

# Source family -> split. A pair's family is the family of its SOURCE, so every
# chapter quoting the same block is evaluated on the same side.
FAMILY_SPLITS = {
    "isaiah-2-14": "dev",
    "isaiah-48-49": "test",
    "isaiah-50-51": "dev",
    "isaiah-52": "test",
    "isaiah-53": "test",
    "isaiah-54": "dev",
    "malachi-3-4": "test",
    "matthew-5-7": "dev",
    "1-corinthians-13": "dev",
    "genesis-2-3": "test",
    "john-17": "test",
    # controls are not source families; split by hand to keep both sides populated
    "control-shared-1": "dev",
    "control-shared-2": "dev",
    "control-shared-3": "test",
    "control-unrelated-1": "dev",
    "control-unrelated-2": "dev",
    "control-unrelated-3": "dev",
    "control-unrelated-4": "dev",
    "control-unrelated-5": "dev",
    "control-unrelated-6": "test",
    "control-unrelated-7": "test",
    "control-unrelated-8": "test",
}

PAIRS = [
    # ---- whole-chapter quotations: Isaiah ---------------------------------
    {"query": "1 Nephi 20", "expected": "Isaiah 48", "kind": "quotation",
     "family": "isaiah-48-49", "confidence": "high"},
    {"query": "1 Nephi 21", "expected": "Isaiah 49", "kind": "quotation",
     "family": "isaiah-48-49", "confidence": "high"},
    {"query": "2 Nephi 7", "expected": "Isaiah 50", "kind": "quotation",
     "family": "isaiah-50-51", "confidence": "high"},
    {"query": "2 Nephi 8", "expected": "Isaiah 51", "kind": "quotation",
     "family": "isaiah-50-51", "confidence": "high"},
    {"query": "2 Nephi 12", "expected": "Isaiah 2", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 13", "expected": "Isaiah 3", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 14", "expected": "Isaiah 4", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 15", "expected": "Isaiah 5", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 16", "expected": "Isaiah 6", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 17", "expected": "Isaiah 7", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 18", "expected": "Isaiah 8", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 19", "expected": "Isaiah 9", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 20", "expected": "Isaiah 10", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 21", "expected": "Isaiah 11", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 22", "expected": "Isaiah 12", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 23", "expected": "Isaiah 13", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "2 Nephi 24", "expected": "Isaiah 14", "kind": "quotation",
     "family": "isaiah-2-14", "confidence": "high"},
    {"query": "Mosiah 14", "expected": "Isaiah 53", "kind": "quotation",
     "family": "isaiah-53", "confidence": "high"},
    {"query": "3 Nephi 22", "expected": "Isaiah 54", "kind": "quotation",
     "family": "isaiah-54", "confidence": "high"},
    {"query": "3 Nephi 24", "expected": "Malachi 3", "kind": "quotation",
     "family": "malachi-3-4", "confidence": "high"},
    {"query": "3 Nephi 25", "expected": "Malachi 4", "kind": "quotation",
     "family": "malachi-3-4", "confidence": "high"},

    # ---- partial quotations ----------------------------------------------
    {"query": "Mosiah 12", "expected": "Isaiah 52", "kind": "partial_quotation",
     "family": "isaiah-52", "query_span": "21-24", "expected_span": "7-10",
     "confidence": "high", "reviewed": True,
     "note": "Mosiah 12:21-24 reproduces Isaiah 52:7-10"},
    {"query": "3 Nephi 20", "expected": "Isaiah 52", "kind": "partial_quotation",
     "family": "isaiah-52", "query_span": "36-40", "expected_span": "1-2,7-10",
     "confidence": "high", "reviewed": True,
     "note": "Isaiah quotation starts at 3 Nephi 20:36 ('Awake, awake'; 'how "
             "beautiful upon the mountains'); vs 23-26 are Moses/Abraham material "
             "and are NOT part of it. The extent of the quotation should still be "
             "confirmed verse by verse."},
    # Re-annotated by review: this is a partial quotation, not a bare control
    {"query": "Moroni 10", "expected": "Isaiah 52", "kind": "partial_quotation",
     "family": "isaiah-52", "query_span": "31", "expected_span": "1-2",
     "confidence": "high", "reviewed": True,
     "note": "Moroni 10:31 echoes Isaiah 52:1-2 ('awake, and arise from the dust', "
             "'put on thy beautiful garments')"},

    # ---- paraphrases ------------------------------------------------------
    {"query": "3 Nephi 12", "expected": "Matthew 5", "kind": "paraphrase",
     "family": "matthew-5-7", "confidence": "high"},
    {"query": "3 Nephi 13", "expected": "Matthew 6", "kind": "paraphrase",
     "family": "matthew-5-7", "confidence": "high"},
    {"query": "3 Nephi 14", "expected": "Matthew 7", "kind": "paraphrase",
     "family": "matthew-5-7", "confidence": "high"},
    {"query": "Moroni 7", "expected": "1 Corinthians 13", "kind": "paraphrase",
     "family": "1-corinthians-13", "query_span": "45-48", "expected_span": "4-8",
     "confidence": "high", "reviewed": True,
     "note": "Moroni 7:45-48 track 1 Cor 13:4-8 with rewording and adaptation"},

    # ---- allusions --------------------------------------------------------
    {"query": "Alma 7", "expected": "Isaiah 53", "kind": "allusion",
     "family": "isaiah-53", "query_span": "10-12", "expected_span": "4",
     "confidence": "low", "reviewed": True,
     "note": "Alma 7:10-12 take up the servant's pains and sicknesses; the link may "
             "be mediated through Matthew 8:17, which applies Isaiah's language to "
             "Jesus"},
    {"query": "3 Nephi 19", "expected": "John 17", "kind": "allusion",
     "family": "john-17", "query_span": "20-23,28-29", "expected_span": "11,20-23",
     "confidence": "medium", "reviewed": True,
     "note": "The prayer's language of being one, the Father in Christ and Christ in "
             "his followers tracks John 17:11, 20-23"},

    # ---- interpretation / commentary -------------------------------------
    # Contextual dependence; scored separately, excluded from retrieval recall.
    {"query": "Mosiah 15", "expected": "Isaiah 53", "kind": "interpretation",
     "family": "isaiah-53", "confidence": "high", "reviewed": True,
     "note": "Abinadi's exposition of the servant passage he has just quoted in "
             "Mosiah 14 — dependence is contextual, so it is not counted as "
             "independent retrieval"},
    {"query": "2 Nephi 2", "expected": "Genesis 3", "kind": "interpretation",
     "family": "genesis-2-3", "query_span": "15-21", "expected_span": "whole",
     "confidence": "medium", "reviewed": True,
     "note": "Lehi retells the Fall (fruit, tree of life, serpent, Eve, Adam, Eden); "
             "the material spans Genesis 2-3, recorded here as Genesis 3"},
    {"query": "Alma 42", "expected": "Genesis 3", "kind": "interpretation",
     "family": "genesis-2-3", "query_span": "2-6", "expected_span": "24",
     "confidence": "medium", "reviewed": True,
     "note": "Alma recounts Eden, the tree of life, the cherubim and the flaming "
             "sword (Genesis 3:24)"},

    # ---- controls ---------------------------------------------------------
    # Only Mosiah 4 has been reviewed; the other two are provisional and stay
    # flagged, so the false-positive estimate is not overstated.
    {"query": "Mosiah 4", "kind": "shared_vocabulary", "family": "control-shared-1",
     "reviewed": True,
     "note": "reviewed: no clear single-chapter KJV source found; kept provisionally "
             "and not treated as a robust control category"},
    {"query": "Alma 32", "kind": "shared_vocabulary", "family": "control-shared-2",
     "reviewed": False,
     "note": "unreviewed: faith discourse, provisional control"},
    {"query": "Helaman 5", "kind": "shared_vocabulary", "family": "control-shared-3",
     "reviewed": False,
     "note": "unreviewed: preaching narrative, provisional control"},

    {"query": "Alma 43", "kind": "unrelated", "family": "control-unrelated-1"},
    {"query": "Alma 50", "kind": "unrelated", "family": "control-unrelated-2"},
    {"query": "Ether 1", "kind": "unrelated", "family": "control-unrelated-3"},
    {"query": "Ether 6", "kind": "unrelated", "family": "control-unrelated-4"},
    {"query": "Moroni 9", "kind": "unrelated", "family": "control-unrelated-5"},
    {"query": "1 Nephi 16", "kind": "unrelated", "family": "control-unrelated-6"},
    {"query": "3 Nephi 4", "kind": "unrelated", "family": "control-unrelated-7"},
    {"query": "Helaman 11", "kind": "unrelated", "family": "control-unrelated-8"},
]

RETRIEVAL_KINDS = ("quotation", "partial_quotation", "paraphrase", "allusion")
CONTEXT_KINDS = ("interpretation",)
CONTROL_KINDS = ("shared_vocabulary", "unrelated")
ALL_KINDS = RETRIEVAL_KINDS + CONTEXT_KINDS + CONTROL_KINDS


def split_for_family(family: str) -> str:
    try:
        return FAMILY_SPLITS[family]
    except KeyError:
        raise KeyError(
            f"family {family!r} has no dev/test assignment; add it to FAMILY_SPLITS"
        ) from None


def all_pairs():
    """Every labelled pair, normalised: kind, family, spans, split, review status."""
    rows = []
    for row in PAIRS:
        record = {
            "query": row["query"],
            "expected": row.get("expected"),
            "kind": row["kind"],
            "family": row["family"],
            "query_span": row.get("query_span", "whole"),
            "expected_span": row.get("expected_span", "whole"),
            "confidence": row.get("confidence", "high"),
            "note": row.get("note", ""),
            "reviewed": bool(row.get("reviewed", False)),
        }
        record["split"] = split_for_family(record["family"])
        rows.append(record)
    return rows


def positives(rows=None):
    """Pairs that assert a relationship (everything but the controls)."""
    rows = rows if rows is not None else all_pairs()
    return [row for row in rows if row["kind"] not in CONTROL_KINDS]


def controls(rows=None):
    rows = rows if rows is not None else all_pairs()
    return [row for row in rows if row["kind"] in CONTROL_KINDS]


def retrieval_positives(rows=None):
    """Positives a retrieval system is expected to find on its own."""
    rows = rows if rows is not None else all_pairs()
    return [row for row in rows if row["kind"] in RETRIEVAL_KINDS]
