"""Labelled cross-corpus pairs for retrieval evaluation.

Each pair says: this chapter of the Book of Mormon is known to reproduce (or
depend on) that chapter of the King James Bible. Quotation pairs are passages
the Book of Mormon explicitly takes over from Isaiah or Malachi; paraphrase
pairs are passages that follow a KJV source closely without copying it word for
word (the Sermon on the Mount material). Controls are chapters with no single
KJV source, used to see what the retriever does when there is nothing to find.

References are chapter-level on both sides. The pairs are standard, checkable
correspondences, not inferences from the model.
"""

QUOTATION_PAIRS = [
    # 1 Nephi 20-21 reproduce Isaiah 48-49
    {"query": "1 Nephi 20", "expected": "Isaiah 48"},
    {"query": "1 Nephi 21", "expected": "Isaiah 49"},
    # 2 Nephi 7-8 reproduce Isaiah 50-51
    {"query": "2 Nephi 7", "expected": "Isaiah 50"},
    {"query": "2 Nephi 8", "expected": "Isaiah 51"},
    # 2 Nephi 12-24 reproduce Isaiah 2-14, chapter for chapter
    {"query": "2 Nephi 12", "expected": "Isaiah 2"},
    {"query": "2 Nephi 13", "expected": "Isaiah 3"},
    {"query": "2 Nephi 14", "expected": "Isaiah 4"},
    {"query": "2 Nephi 15", "expected": "Isaiah 5"},
    {"query": "2 Nephi 16", "expected": "Isaiah 6"},
    {"query": "2 Nephi 17", "expected": "Isaiah 7"},
    {"query": "2 Nephi 18", "expected": "Isaiah 8"},
    {"query": "2 Nephi 19", "expected": "Isaiah 9"},
    {"query": "2 Nephi 20", "expected": "Isaiah 10"},
    {"query": "2 Nephi 21", "expected": "Isaiah 11"},
    {"query": "2 Nephi 22", "expected": "Isaiah 12"},
    {"query": "2 Nephi 23", "expected": "Isaiah 13"},
    {"query": "2 Nephi 24", "expected": "Isaiah 14"},
    # Mosiah 14 reproduces Isaiah 53
    {"query": "Mosiah 14", "expected": "Isaiah 53"},
    # 3 Nephi 22 reproduces Isaiah 54
    {"query": "3 Nephi 22", "expected": "Isaiah 54"},
    # 3 Nephi 24-25 reproduce Malachi 3-4
    {"query": "3 Nephi 24", "expected": "Malachi 3"},
    {"query": "3 Nephi 25", "expected": "Malachi 4"},
]

PARAPHRASE_PAIRS = [
    # 3 Nephi 12-14 follow the Sermon on the Mount closely but with edits
    {"query": "3 Nephi 12", "expected": "Matthew 5"},
    {"query": "3 Nephi 13", "expected": "Matthew 6"},
    {"query": "3 Nephi 14", "expected": "Matthew 7"},
]

CONTROLS = [
    # war narrative, no single KJV source
    {"query": "Alma 43"},
    # genealogy, structurally similar to KJV genealogies without quoting one
    {"query": "Ether 1"},
    # a captain Moroni epistle, no KJV source
    {"query": "Moroni 9"},
]


def all_pairs():
    """Return every labelled pair with its kind; controls carry expected=None."""
    pairs = []
    for row in QUOTATION_PAIRS:
        pairs.append({**row, "kind": "quotation"})
    for row in PARAPHRASE_PAIRS:
        pairs.append({**row, "kind": "paraphrase"})
    for row in CONTROLS:
        pairs.append({"query": row["query"], "expected": None, "kind": "control"})
    return pairs
