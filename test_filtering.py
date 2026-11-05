# test_filtering.py
"""Tests for n-gram statistics and the filtering decision.

The two statistics are deliberately different:

  occurrence count    every position across every window, so overlapping windows
                      and repeated phrases inflate it
  document frequency  how many chapters contain the n-gram at least once
"""

import pytest

from filtering import (
    build_ngram_document_frequencies,
    build_ngram_frequencies,
    identify_high_frequency_ngrams,
    identify_ngrams_by_document_frequency,
    identify_ngrams_by_occurrence,
    remove_high_freq_phrases,
)


def window(tokens, book="1 Nephi", chapter=1, corpus="bom"):
    return {
        "corpus": corpus,
        "book": book,
        "chapter": chapter,
        "tokens": tokens,
    }


def test_occurrence_count_is_inflated_by_overlapping_windows():
    """Same passage, three overlapping windows: occurrences triple, documents stay 1."""
    phrase = ["and", "it", "came", "to", "pass", "that"]
    windows = [
        window(phrase + ["alpha"]),
        window(["x"] + phrase + ["beta"]),
        window(["y", "z"] + phrase),
    ]

    occurrences = build_ngram_frequencies(windows, n=3)
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    assert occurrences[("and", "it", "came")] == 3
    assert documents[("and", "it", "came")] == 1
    assert total_documents == 1


def test_repeated_phrase_within_one_chapter_counts_once_per_document():
    phrase = ["the", "lord", "god"]
    windows = [
        window(phrase + phrase + phrase),
        window(phrase + ["other", "words"]),
    ]

    occurrences = build_ngram_frequencies(windows, n=3)
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    assert occurrences[("the", "lord", "god")] == 4  # 3 inside the first window + 1
    assert documents[("the", "lord", "god")] == 1
    assert total_documents == 1


def test_document_frequency_counts_chapters_not_windows():
    phrase = ["and", "it", "came", "to", "pass"]
    windows = [
        # chapter 1, two overlapping windows
        window(["a"] + phrase, book="1 Nephi", chapter=1),
        window(phrase + ["b"], book="1 Nephi", chapter=1),
        # chapter 2, contains it
        window(["c"] + phrase, book="1 Nephi", chapter=2),
        # chapter 3, does not
        window(["d", "e", "f", "g", "h"], book="1 Nephi", chapter=3),
    ]

    occurrences = build_ngram_frequencies(windows, n=3)
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    assert total_documents == 3
    assert documents[("and", "it", "came")] == 2     # two chapters, not three windows
    assert occurrences[("and", "it", "came")] == 3   # occurrences counted per window


def test_same_book_and_chapter_in_different_corpora_are_two_documents():
    phrase = ["and", "it", "came", "to", "pass"]
    windows = [
        window(phrase, corpus="bom", book="Genesis", chapter=1),
        window(phrase, corpus="kjv", book="Genesis", chapter=1),
    ]

    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    assert total_documents == 2
    assert documents[("and", "it", "came")] == 2


def test_document_frequency_threshold_is_a_fraction_of_documents():
    # four documents: phrase A in 2 of 4 (50%), phrase B in 1 of 4 (25%)
    windows = [
        window(["aaa", "bbb", "ccc", "ddd"], chapter=1),
        window(["aaa", "bbb", "ccc", "eee"], chapter=2),
        window(["fff", "ggg", "hhh", "iii"], chapter=3),
        window(["jjj", "kkk", "lll", "mmm"], chapter=4),
    ]
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    selected = identify_ngrams_by_document_frequency(
        documents, total_documents=total_documents, min_document_frequency=0.4
    )
    assert ("aaa", "bbb", "ccc") in selected       # 0.5 > 0.4
    assert ("fff", "ggg", "hhh") not in selected   # 0.25

    strict = identify_ngrams_by_document_frequency(
        documents, total_documents=total_documents, min_document_frequency=0.5
    )
    assert ("aaa", "bbb", "ccc") not in strict     # threshold is exclusive


def test_occurrence_selector_uses_an_absolute_count():
    occurrences = build_ngram_frequencies(
        [window(["a", "b", "c"]), window(["a", "b", "c"]), window(["x", "y", "z"])], n=3
    )

    assert identify_ngrams_by_occurrence(occurrences, min_occurrence_count=2) == {
        ("a", "b", "c")
    }
    assert identify_ngrams_by_occurrence(occurrences, min_occurrence_count=1) == {
        ("a", "b", "c"),
        ("x", "y", "z"),
    }


def test_legacy_selector_reproduces_occurrence_rate_behaviour():
    occurrences = build_ngram_frequencies(
        [window(["a", "b", "c"]), window(["a", "b", "c"]), window(["x", "y", "z"])], n=3
    )

    # 2 occurrences / 3 windows = 0.667 > 0.5
    assert identify_high_frequency_ngrams(occurrences, total_windows=3,
                                          frequency_threshold=0.5) == {("a", "b", "c")}


@pytest.mark.parametrize(
    "call",
    [
        lambda oc, df: identify_ngrams_by_document_frequency(df, 0, 0.01),
        lambda oc, df: identify_ngrams_by_document_frequency(df, 4, -0.1),
        lambda oc, df: identify_ngrams_by_document_frequency(df, 4, 1.1),
        lambda oc, df: identify_ngrams_by_occurrence(oc, 0),
        lambda oc, df: identify_high_frequency_ngrams(oc, 0, 0.01),
    ],
)
def test_invalid_thresholds_are_rejected(call):
    occurrences = build_ngram_frequencies([window(["a", "b", "c"])], n=3)
    documents, total_documents = build_ngram_document_frequencies(
        [window(["a", "b", "c"])], n=3
    )
    with pytest.raises(ValueError):
        call(occurrences, documents)


def test_remove_high_freq_phrases_strips_selected_trigrams():
    windows = [window(["and", "it", "came", "to", "pass", "that", "he", "spake"])]
    selected = {("and", "it", "came"), ("to", "pass", "that")}

    out = remove_high_freq_phrases(windows, selected, n=3)

    assert out[0]["filtered_tokens"] == ["he", "spake"]
    assert out[0]["filtered_text"] == "he spake"
    # the original tokens are preserved for later statistics
    assert out[0]["tokens"] == ["and", "it", "came", "to", "pass", "that", "he", "spake"]


def test_real_corpus_thresholds_are_not_comparable_across_metrics():
    """The two measures have different denominators, and it matters a lot.

    1% of 239 chapters means "in 3 chapters"; 1% of ~17.8k windows means
    "observed 178 times". Using the same number for both is not a like-for-like
    comparison, which is why the pipeline calibrates each metric separately.
    """
    from pathlib import Path

    from chunking import build_chapter_token_stream, create_overlapping_windows
    from preprocess import preprocess_bom

    candidates = [Path(__file__).parent / "data" / "bom.txt",
                  Path(__file__).parent / "bom.txt"]
    corpus = next((p for p in candidates if p.exists()), None)
    if corpus is None:
        pytest.skip("no local corpus to check against")

    windows = create_overlapping_windows(
        build_chapter_token_stream(preprocess_bom(str(corpus))),
        window_size=30, overlap=15,
    )

    occurrences = build_ngram_frequencies(windows, n=3)
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)

    by_document = identify_ngrams_by_document_frequency(documents, total_documents, 0.01)
    by_occurrence = identify_high_frequency_ngrams(
        occurrences, total_windows=len(windows), frequency_threshold=0.01
    )

    assert total_documents == 239
    assert by_occurrence != by_document
    # the document scale at 1% is far looser: it selects two orders of magnitude more
    assert len(by_document) > 10 * len(by_occurrence)


def test_real_corpus_default_document_threshold_is_calibrated():
    """The pipeline default must filter a sane share of the text, not half of it."""
    from pathlib import Path

    from chunking import build_chapter_token_stream, create_overlapping_windows
    from main import MIN_DOCUMENT_FREQUENCY
    from preprocess import preprocess_bom

    candidates = [Path(__file__).parent / "data" / "bom.txt",
                  Path(__file__).parent / "bom.txt"]
    corpus = next((p for p in candidates if p.exists()), None)
    if corpus is None:
        pytest.skip("no local corpus to check against")

    assert 0.10 <= MIN_DOCUMENT_FREQUENCY <= 0.25

    windows = create_overlapping_windows(
        build_chapter_token_stream(preprocess_bom(str(corpus))),
        window_size=30, overlap=15,
    )
    documents, total_documents = build_ngram_document_frequencies(windows, n=3)
    selected = identify_ngrams_by_document_frequency(
        documents, total_documents, MIN_DOCUMENT_FREQUENCY
    )

    total_tokens = sum(len(w["tokens"]) for w in windows)
    stripped = [dict(tokens=w["tokens"]) for w in windows]
    remove_high_freq_phrases(stripped, selected, n=3)
    removed = total_tokens - sum(len(w["filtered_tokens"]) for w in stripped)

    share = removed / total_tokens
    assert 0.03 <= share <= 0.20, f"default threshold strips {share:.1%} of tokens"

