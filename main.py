from preprocess import preprocess_bom
from chunking import build_chapter_token_stream, create_overlapping_windows
from filtering import (
    build_ngram_frequencies,
    identify_high_frequency_ngrams,
    remove_high_freq_phrases
)
from similarity import (
    add_minhashes,
    build_lsh,
    find_minhash_matches,
    add_embeddings,
    build_faiss_index,
    find_embedding_matches,
    filter_nearby_matches,
    get_top_matches
)
import pandas as pd


# --------------------------------------------
# 1. PREPROCESS
# --------------------------------------------

print("Preprocessing...")
bom_data = preprocess_bom("bom.txt", remove_punctuation=False)


# --------------------------------------------
# 2. CHUNKING
# --------------------------------------------

print("Creating overlapping windows...")
chapter_streams = build_chapter_token_stream(bom_data)
windows = create_overlapping_windows(
    chapter_streams,
    window_size=30,
    overlap=15
)


# --------------------------------------------
# 3. REMOVE HIGH-FREQUENCY PHRASES
# --------------------------------------------

print("Computing n-gram frequencies...")
ngram_counts = build_ngram_frequencies(windows, n=3)

high_freq = identify_high_frequency_ngrams(
    ngram_counts,
    total_windows=len(windows),
    frequency_threshold=0.01
)

print(f"High-frequency phrases identified: {len(high_freq)}")

windows = remove_high_freq_phrases(
    windows,
    high_freq,
    n=3
)


# --------------------------------------------
# 4. MINHASH MATCHING
# --------------------------------------------

print("Running MinHash matching...")
windows = add_minhashes(windows)

lsh = build_lsh(windows, threshold=0.6)

minhash_matches = find_minhash_matches(windows, lsh)

print(f"Raw MinHash matches: {len(minhash_matches)}")


# --------------------------------------------
# 5. EMBEDDING MATCHING
# --------------------------------------------

print("Computing embeddings...")
windows = add_embeddings(windows)

index, embeddings = build_faiss_index(windows)

embedding_matches = find_embedding_matches(
    windows,
    index,
    embeddings,
    k=5
)

print(f"Raw embedding matches: {len(embedding_matches)}")


# --------------------------------------------
# 6. MERGE + FILTER
# --------------------------------------------

all_matches = minhash_matches + embedding_matches

print("Filtering nearby trivial matches...")
filtered_matches = filter_nearby_matches(
    windows,
    all_matches,
    min_token_distance=200
)

print(f"Filtered matches: {len(filtered_matches)}")


# --------------------------------------------
# 7. SCORE + OUTPUT CSV
# --------------------------------------------
print("Scoring matches and writing CSV...")

top_matches = get_top_matches(
    windows,
    filtered_matches,
    top_n=750
)

# Prepare rows for CSV
rows = []
for score, i, j, method in top_matches:
    w1 = windows[i]
    w2 = windows[j]
    
    rows.append({
        "score": score,
        "method": method,
        "text_1": w1["text"],
        "book_1": w1["book"],
        "chapter_1": w1["chapter"],
        "start_verse_1": w1["start_verse"],
        "end_verse_1": w1["end_verse"],
        "text_2": w2["text"],
        "book_2": w2["book"],
        "chapter_2": w2["chapter"],
        "start_verse_2": w2["start_verse"],
        "end_verse_2": w2["end_verse"],
    })

# Create DataFrame and save CSV
df = pd.DataFrame(rows)
df.to_csv("top_matches.csv", index=False)
print("CSV saved as top_matches.csv")