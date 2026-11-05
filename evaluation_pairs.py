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
    "transfiguration": "test",
    "psalm-85": "test",
    "exodus-20-decalogue": "test",
    "acts-2-pentecost": "test",
    "abrahamic-promise": "test",
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
    "control-unrelated-1": "dev",
    "control-unrelated-2": "dev",
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
     "note": "Mosiah 12:21-24 reproduces Isaiah 52:7-10",
     "also_expected": [
         {"ref": "Exodus 20", "span": "2-4", "query_span": "34-36",
          "family": "exodus-20-decalogue",
          "kind": "quotation", "reviewed": True, "confidence": "high",
          "note": "a SECOND, independent quotation in the same chapter: Mosiah "
                  "12:34-36 reproduces Exodus 20:2-4 (the Decalogue). Crediting only "
                  "one source scores a correct retrieval as a miss."},
     ]},
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

    # ---- reviewed from the text ------------------------------------------
    {"query": "Helaman 5", "expected": "Luke 9", "kind": "allusion",
     "family": "transfiguration", "query_span": "28-29", "expected_span": "34-35",
     "confidence": "low", "reviewed": True,
     "note": "reviewed: the cloud that overshadows and the voice out of it pattern on "
             "the transfiguration ('there came a cloud, and overshadowed them ... a "
             "voice out of the cloud'); best 0.7975, reworded. Low confidence: the "
             "scene is patterned, the wording is not reproduced.",
     "also_expected": [
         {"ref": "Acts 2", "span": "3-4", "query_span": "45",
          "family": "acts-2-pentecost", "kind": "allusion",
          "confidence": "low", "reviewed": True,
          "note": "the Spirit descending as fire, and speech following, patterns on "
                  "Pentecost; best 0.7754"},
     ]},
    {"query": "Ether 1", "expected": "Genesis 26", "kind": "allusion",
     "family": "abrahamic-promise", "query_span": "43", "expected_span": "3-4",
     "confidence": "low", "reviewed": True,
     "note": "reviewed, reconsidered twice: 'I will bless thee and thy seed' plus a "
             "promise to raise up a great nation follows the Abrahamic covenant formula "
             "of Genesis 26:3-4 (land, multiplied seed, all nations blessed). The wording "
             "is not reproduced and the real source family is the Abrahamic promises "
             "rather than this chapter alone, so: low-confidence allusion, not a "
             "quotation, and no longer a clean negative control."},
    {"query": "Helaman 11", "expected": "Psalms 85", "kind": "allusion",
     "family": "psalm-85", "query_span": "11-12", "expected_span": "4-5",
     "confidence": "low", "reviewed": True,
     "note": "reviewed: Nephi's plea that God turn away his anger follows Psalm 85's "
             "prayer ('cause thine anger toward us to cease ... wilt thou be angry with "
             "us for ever'); 27 window matches, best 0.8353. A prayer formula, not a "
             "quotation."},

    # ---- controls ---------------------------------------------------------
    # Every control here was checked against the text: the top candidate chapters
    # and the actual matched wording were read. None reproduces a source closely
    # enough to be a quotation, and the closest approach is recorded so the
    # strength of each control is visible rather than assumed. Two chapters that
    # began life here (Helaman 5, Helaman 11) were moved out above: they pattern
    # on a specific passage even though they do not quote it.
    {"query": "Mosiah 4", "kind": "shared_vocabulary", "family": "control-shared-1",
     "reviewed": True,
     "note": "reviewed: forgiveness and charity language; closest 2 Corinthians 2:9-10 "
             "(0.7601) and James 2:5-6 (0.7484). No reproduced wording."},
    {"query": "Alma 32", "kind": "shared_vocabulary", "family": "control-shared-2",
     "reviewed": True,
     "note": "reviewed: faith discourse. Closest Romans 9:17-19 (0.7523), John 6:29-30 "
             "(0.7507, the demand for a sign), James 5:7-8 (0.7497, patience for the "
             "fruit). Conceptual parallels inside a sermon, no source wording."},
    {"query": "Alma 43", "kind": "shared_vocabulary", "family": "control-unrelated-1",
     "reviewed": True,
     "note": "reviewed: Alma 43:46 adapts the individual-responsibility rule of "
             "Deuteronomy 24:16 ('not guilty of the first offense' / 'the fathers shall "
             "not be put to death for the children') as a legal principle rather than "
             "quoting it; best 0.7285."},
    {"query": "Alma 50", "kind": "shared_vocabulary", "family": "control-unrelated-2",
     "reviewed": True,
     "note": "reviewed: the covenant 'inasmuch as they keep my commandments they shall "
             "prosper in the land' formula, closest Deuteronomy 33:9-10 (0.7277); "
             "inherited covenant language, not a source quotation."},
    {"query": "Ether 6", "kind": "shared_vocabulary", "family": "control-unrelated-4",
     "reviewed": True,
     "note": "reviewed: the storm-driven voyage (Jonah 1:3-4, 0.7323) and the "
             "'choose ye out from among our sons a king' formula (Deuteronomy 17:15, "
             "0.7537). Shared narrative and legal motifs, no reproduced wording."},
    {"query": "Moroni 9", "kind": "shared_vocabulary", "family": "control-unrelated-5",
     "reviewed": True,
     "note": "reviewed: the epistolary opening 'my son, be faithful in Christ' echoes "
             "2 Timothy 2:1 (0.7596); affliction language echoes Psalm 107:17-19 "
             "(0.7356). Generic letter and lament formulae."},
    {"query": "1 Nephi 16", "kind": "shared_vocabulary", "family": "control-unrelated-6",
     "reviewed": True,
     "note": "reviewed: wilderness travel 'by the way of the Red Sea' echoes "
             "Deuteronomy 2:1 (0.7713) and Numbers 27:3 (0.7251). Travel formula, "
             "no reproduced wording."},
    {"query": "3 Nephi 4", "kind": "shared_vocabulary", "family": "control-unrelated-7",
     "reviewed": True,
     "note": "reviewed: Giddianhi's corpse 'hanged upon a tree' until dead touches the "
             "Deuteronomy 21:22-23 legal wording (0.7757) and Acts 10:39 (0.7724), and "
             "the action follows that law; the wording does not. Second strongest "
             "control."},
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

        targets = []
        if record["expected"]:
            targets.append({
                "ref": record["expected"],
                "span": record["expected_span"],
                "kind": record["kind"],
                "confidence": record["confidence"],
                "reviewed": record["reviewed"],
                "note": record["note"],
                "query_span": record["query_span"],
                "family": record["family"],
                "primary": True,
            })
        for extra in row.get("also_expected", []):
            targets.append({
                "ref": extra["ref"],
                "span": extra.get("span", "whole"),
                "kind": extra.get("kind", record["kind"]),
                "confidence": extra.get("confidence", "medium"),
                "reviewed": bool(extra.get("reviewed", False)),
                "note": extra.get("note", ""),
                "query_span": extra.get("query_span", record["query_span"]),
                "family": extra.get("family", record["family"]),
                "primary": False,
            })
        record["expected_targets"] = targets

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
