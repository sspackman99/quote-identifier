# demo_top_5_candidates.py

"""
Demo script to verify top-5 candidate retrieval with real BOM data.
"""

from preprocess import preprocess_bom
from chunking import build_chapter_token_stream, create_overlapping_windows
from filtering import (
    build_ngram_frequencies,
    identify_high_frequency_ngrams,
    remove_high_freq_phrases
)
from similarity import (
    add_minhashes,
    add_embeddings,
    build_faiss_index,
    filter_nearby_matches
)
import top_5_candidates


def main():
    """Run demo with real BOM data."""
    print("=" * 60)
    print("Top-5 Cosine Candidate Retrieval Demo")
    print("=" * 60)
    
    # 1. Preprocess
    print("\n[1/6] Preprocessing BOM...")
    bom_data = preprocess_bom("bom.txt", remove_punctuation=False)
    
    # 2. Chunking
    print("[2/6] Creating overlapping windows...")
    chapter_streams = build_chapter_token_stream(bom_data)
    windows = create_overlapping_windows(
        chapter_streams,
        window_size=30,
        overlap=15
    )
    print(f"      Created {len(windows)} windows")
    
    # 3. Remove high-frequency phrases
    print("[3/6] Removing high-frequency phrases...")
    ngram_counts = build_ngram_frequencies(windows, n=3)
    high_freq = identify_high_frequency_ngrams(
        ngram_counts,
        total_windows=len(windows),
        frequency_threshold=0.01
    )
    print(f"      High-frequency phrases: {len(high_freq)}")
    windows = remove_high_freq_phrases(windows, high_freq, n=3)
    print(f"      Windows after filtering: {len(windows)}")
    
    # 4. Add embeddings
    print("[4/6] Computing embeddings...")
    windows = add_embeddings(windows)
    print(f"      Embedding dimension: {len(windows[0]['embedding'])}")
    
    # 5. Build FAISS index
    print("[5/6] Building FAISS index...")
    index, embeddings = build_faiss_index(windows)
    
    # 6. Generate top-5 candidates
    print("[6/6] Generating top-5 cosine candidates...")
    candidates = top_5_candidates.find_top_5_embedding_candidates(windows, index, embeddings)
    print(f"      Generated {len(candidates)} candidate records")
    
    # Export to CSV
    csv_path = "top_5_embedding_candidates.csv"
    top_5_candidates.export_top_5_candidates_to_csv(candidates, csv_path)
    print(f"      Saved to {csv_path}")
    
    # Demo: Show sample candidates
    print("\n" + "=" * 60)
    print("Sample Candidates (first 5 sources, rank 1-2):")
    print("=" * 60)
    
    source_ids = set()
    sample_count = 0
    max_per_source = 2
    
    for c in candidates:
        if c["source_window_id"] not in source_ids and sample_count < 10:
            print(f"\nSource: {c['source_book']} {c['source_chapter']}:{c['source_start_verse']}-{c['source_end_verse']}")
            print(f"  Text: {c['source_text'][:100]}...")
            print(f"  Rank {c['candidate_rank']}: {c['candidate_book']} {c['candidate_chapter']}:{c['candidate_start_verse']}-{c['candidate_end_verse']}")
            print(f"  Cosine similarity: {c['cosine_similarity']:.6f}")
            print(f"  Text: {c['candidate_text'][:100]}...")
            print("-" * 60)
            
            source_ids.add(c["source_window_id"])
            sample_count += 1
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
