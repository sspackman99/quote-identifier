# top_5_candidates.py

from similarity import build_faiss_index
import numpy as np


def find_top_5_embedding_candidates(windows, index, embeddings):
    """
    For each source window, find up to 5 candidates ranked by cosine similarity.
    
    Args:
        windows: List of window dicts with embeddings and metadata (book, chapter, 
                start_token_index, text, etc.)
        index: FAISS index
        embeddings: NumPy array of embeddings (shape: [num_windows, dim])
    
    Returns:
        List of candidate records with full metadata and cosine similarity.
        Each record contains: source_window_id, source_book, source_chapter,
        source_start_verse, source_end_verse, source_start_token_index, source_text,
        candidate_rank, candidate_window_id, candidate_book, candidate_chapter,
        candidate_start_verse, candidate_end_verse, candidate_start_token_index,
        candidate_text, cosine_similarity
    """
    candidates = []
    
    # k needs to be large enough to get at least 5 non-self matches after exclusions
    # Using len(windows) - 1 to avoid self-match, then we'll filter to top 5
    k = min(len(windows) - 1, 50)
    
    for i, emb in enumerate(embeddings):
        # Search for nearest neighbors
        D, I = index.search(np.array([emb]), k)
        
        # Sort by cosine similarity (descending), then by window index (ascending) for tie-breaking
        # D is negative distance for IndexFlatIP (higher inner product = smaller negative)
        # So we sort by -D[0] ascending, then I ascending
        sorted_indices = sorted(enumerate(I[0]), key=lambda x: (-x[1][0], x[0]))
        
        for rank, j in sorted_indices[:5]:
            # Skip self-match
            if j == i:
                continue
            
            # Apply near-overlap filtering
            w1, w2 = windows[i], windows[j]
            if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
                distance = abs(w1["start_token_index"] - w2["start_token_index"])
                if distance < 200:
                    continue
            
            # Cosine similarity is the inner product (embeddings are normalized)
            # FAISS returns negative distance for IndexFlatIP, so negate it
            cos_sim = float(-D[0][j])
            
            # Build candidate record
            candidate_record = {
                "source_window_id": i,
                "source_book": w1["book"],
                "source_chapter": w1["chapter"],
                "source_start_verse": w1["start_verse"],
                "source_end_verse": w1["end_verse"],
                "source_start_token_index": w1["start_token_index"],
                "source_text": w1["text"],
                "candidate_rank": rank + 1,
                "candidate_window_id": j,
                "candidate_book": w2["book"],
                "candidate_chapter": w2["chapter"],
                "candidate_start_verse": w2["start_verse"],
                "candidate_end_verse": w2["end_verse"],
                "candidate_start_token_index": w2["start_token_index"],
                "candidate_text": w2["text"],
                "cosine_similarity": cos_sim,
            }
            candidates.append(candidate_record)
    
    return candidates


def export_top_5_candidates_to_csv(candidates, filepath="top_5_embedding_candidates.csv"):
    """
    Export top-5 candidates to CSV format.
    
    Args:
        candidates: List of candidate records from find_top_5_embedding_candidates
        filepath: Output CSV path
    """
    import pandas as pd
    
    # Select columns in order
    fieldnames = [
        "source_window_id", "source_book", "source_chapter", 
        "source_start_verse", "source_end_verse", "source_start_token_index", 
        "source_text", "candidate_rank", "candidate_window_id", "candidate_book",
        "candidate_chapter", "candidate_start_verse", "candidate_end_verse",
        "candidate_start_token_index", "candidate_text", "cosine_similarity"
    ]
    
    df = pd.DataFrame(candidates)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(candidates)} candidate records to {filepath}")
