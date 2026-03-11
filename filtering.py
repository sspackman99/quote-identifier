import re
from collections import Counter
from typing import List


# -------------------------------------------------
# 1. Build N-gram Frequency Table
# -------------------------------------------------

def build_ngram_frequencies(windows: List[dict], n=3):
    """
    Builds global n-gram frequency counts across all windows.
    """
    
    counter = Counter()
    
    for window in windows:
        tokens = window["tokens"]
        
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            counter[ngram] += 1
    
    return counter


# -------------------------------------------------
# 2. Identify High-Frequency N-grams
# -------------------------------------------------

def identify_high_frequency_ngrams(counter: Counter,
                                    total_windows: int,
                                    frequency_threshold=0.01):
    """
    Identify n-grams that appear in more than X% of windows.
    
    frequency_threshold=0.01 means 1% of all windows.
    """
    
    high_freq = set()
    
    for ngram, count in counter.items():
        if count / total_windows > frequency_threshold:
            high_freq.add(ngram)
    
    return high_freq


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