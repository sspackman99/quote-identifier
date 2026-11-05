import re
from collections import Counter
from typing import Dict, List, Set, Tuple


# -------------------------------------------------
# 1. N-gram statistics
# -------------------------------------------------
#
# Two different numbers, deliberately kept apart:
#
#   occurrence count   every position an n-gram appears at, across every window.
#                      Windows overlap and a phrase can repeat inside one passage,
#                      so this is inflated relative to how usual the phrase is.
#   document frequency how many DOCUMENTS (chapters) contain the n-gram at least
#                      once. A document is keyed by (corpus, book, chapter), so
#                      the same book/chapter in a different corpus is a different
#                      document.
#
# The filtering decision is made from document frequency; occurrence counts stay
# available so the two can be compared on the same corpus.

def build_ngram_frequencies(windows: List[dict], n=3):
    """Raw OCCURRENCE counts: every position, across all windows, counted each time.

    Overlapping windows and repeated phrases both inflate this number, so it
    measures how often the text repeats itself, not how many passages use it.
    """
    counter = Counter()

    for window in windows:
        tokens = window["tokens"]

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            counter[ngram] += 1

    return counter


def build_ngram_document_frequencies(windows: List[dict], n=3):
    """DOCUMENT frequency: number of distinct (corpus, book, chapter) documents
    whose text contains the n-gram at least once."""
    documents: Dict[Tuple[str, str, int], Set[Tuple[str, ...]]] = {}

    for window in windows:
        key = (window["corpus"], window["book"], window["chapter"])
        ngrams = documents.setdefault(key, set())

        tokens = window["tokens"]
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i:i+n]))

    frequency = Counter()
    for ngrams in documents.values():
        frequency.update(ngrams)

    return frequency, len(documents)


# -------------------------------------------------
# 2. Select frequent n-grams (explicit threshold units)
# -------------------------------------------------

def identify_ngrams_by_document_frequency(document_frequency: Counter,
                                          total_documents: int,
                                          min_document_frequency: float = 0.01):
    """Select n-grams present in more than `min_document_frequency` of documents.

    min_document_frequency is a FRACTION OF DOCUMENTS (0.01 = 1% of chapters),
    not a fraction of windows and not a raw count.
    """
    if total_documents <= 0:
        raise ValueError("total_documents must be positive")
    if not 0.0 <= min_document_frequency <= 1.0:
        raise ValueError("min_document_frequency must be between 0.0 and 1.0")

    return {
        ngram
        for ngram, count in document_frequency.items()
        if count / total_documents > min_document_frequency
    }


def identify_ngrams_by_occurrence(occurrence_counts: Counter,
                                  min_occurrence_count: int):
    """Select n-grams seen at least `min_occurrence_count` times anywhere.

    An absolute COUNT of positions, not a rate — the unit is in the name.
    """
    if min_occurrence_count < 1:
        raise ValueError("min_occurrence_count must be at least 1")

    return {
        ngram
        for ngram, count in occurrence_counts.items()
        if count >= min_occurrence_count
    }


def identify_high_frequency_ngrams(counter: Counter,
                                   total_windows: int,
                                   frequency_threshold=0.01):
    """LEGACY occurrence-rate selector: count / total_windows.

    Kept so the previous behaviour can be reproduced and compared. The ratio
    mixes occurrences with a window count, so it is not a document frequency;
    prefer identify_ngrams_by_document_frequency for filtering decisions.
    """
    if total_windows <= 0:
        raise ValueError("total_windows must be positive")

    return {
        ngram
        for ngram, count in counter.items()
        if count / total_windows > frequency_threshold
    }


# -------------------------------------------------
# 3. Remove High-Frequency N-grams from Windows
# -------------------------------------------------

def remove_high_freq_phrases(windows: List[dict],
                             high_freq_ngrams: set,
                             n=3):
    """
    Removes high-frequency n-grams from window tokens.
    """
    
    for window in windows:
        tokens = window["tokens"]
        i = 0
        filtered_tokens = []
        
        while i < len(tokens):
            match = False
            
            if i <= len(tokens) - n:
                candidate = tuple(tokens[i:i+n])
                
                if candidate in high_freq_ngrams:
                    i += n
                    match = True
            
            if not match:
                filtered_tokens.append(tokens[i])
                i += 1
        
        window["filtered_tokens"] = filtered_tokens
        window["filtered_text"] = " ".join(filtered_tokens)
    
    return windows


# -------------------------------------------------
# Example usage
# -------------------------------------------------

if __name__ == "__main__":
    print("This file is meant to be imported, not run directly.")
