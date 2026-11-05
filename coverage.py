"""Measure how much of a source passage is actually reproduced, verse by verse.

The rubric's extent rule - is the whole passage quoted, or an excerpt - has to be
measurable, otherwise every classifier call is a judgement about coverage with no
evidence behind it. Embedding matches cannot supply that measurement: retrieval matches
many weak windows spread across a chapter, so chapter-level coverage saturates at 1.00
even when three verses are quoted.

This module answers the question with wording instead of similarity. For a source
chapter it asks, verse by verse, what fraction of that verse's four-word sequences also
occur in the query chapter, and reports the pattern:

    reproduced_share      share of the source chapter's verses that are reproduced
    longest_run           longest run of consecutive reproduced verses
    query_share           share of the query's four-word sequences that come from the source

Mosiah 14 -> Isaiah 53 reproduces the whole chapter; Moroni 10:31 -> Isaiah 52:1-2
reproduces two verses out of fifteen; Helaman 11 -> Psalm 85 reproduces nothing.
"""

import re

NGRAM_N = 4
REPRODUCED_VERSE_THRESHOLD = 0.50
WHOLE_PASSAGE_SHARE = 0.60


def tokenise(text):
    """Lowercase word tokens. Punctuation and italic markup do not count."""
    return re.findall(r"[a-z]+", str(text).lower())


def ngram_set(tokens, n=NGRAM_N):
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def verse_containment(verse_text, query_ngrams):
    """Fraction of this verse's n-grams that also occur in the query."""
    grams = ngram_set(tokenise(verse_text))
    if not grams:
        return None
    return len(grams & query_ngrams) / len(grams)


def measure(source_verses, query_text):
    """Measure reproduction between one source chapter and one query passage.

    source_verses: ordered [(verse_number, text), ...] for the source chapter
    query_text:   the text of the query passage (chapter or span)
    """
    if not source_verses:
        raise ValueError("no source verses to measure")

    query_tokens = tokenise(query_text)
    query_ngrams = ngram_set(query_tokens)
    source_ngrams = set()
    per_verse = {}

    for number, text in source_verses:
        grams = ngram_set(tokenise(text))
        source_ngrams |= grams
        containment = verse_containment(text, query_ngrams)
        per_verse[number] = containment if containment is not None else 0.0

    reproduced = [number for number, score in per_verse.items()
                  if score >= REPRODUCED_VERSE_THRESHOLD]

    longest_run = 0
    current_run = 0
    for number in sorted(per_verse):
        if number in reproduced:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    query_grams = ngram_set(query_tokens)
    query_share = (len(query_grams & source_ngrams) / len(query_grams)) if query_grams else 0.0

    return {
        "verses_in_source": len(source_verses),
        "reproduced_verses": reproduced,
        "reproduced_count": len(reproduced),
        "reproduced_share": len(reproduced) / len(source_verses),
        "longest_run": longest_run,
        "query_share": query_share,
        "per_verse": per_verse,
    }


def extent_reading(measurement):
    """Turn a measurement into a rubric reading, with the evidence that produced it.

    Deterministic and conservative: it declines to answer when the measurement is
    ambiguous rather than guessing.
    """
    share = measurement["reproduced_share"]
    run = measurement["longest_run"]
    count = measurement["reproduced_count"]

    if count == 0:
        return {
            "extent": "no reproduction",
            "reason": "no source verse reaches the reproduction threshold",
        }

    if share >= WHOLE_PASSAGE_SHARE:
        return {
            "extent": "quotation",
            "reason": f"{count} of {measurement['verses_in_source']} source verses "
                      f"reproduced ({share:.0%}), run of {run} - the passage as a whole",
        }

    return {
        "extent": "partial_quotation",
        "reason": f"{count} of {measurement['verses_in_source']} source verses "
                  f"reproduced ({share:.0%}), run of {run} - an excerpt of the source",
    }
