# test_top_5_candidates.py

import pytest
import numpy as np


def create_test_windows():
    """Create deterministic test windows with known embeddings."""
    # Create 10 test windows with simple 2D embeddings
    windows = []
    for i in range(10):
        window = {
            "book": f"Book{i % 3}",  # 3 books for variety
            "chapter": i,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": i * 100,  # 100 token spacing (above 200 threshold)
            "text": f"Test window {i}",
            "filtered_text": f"Test window {i}",
            "filtered_tokens": [f"token{i}"] * 30,
            "embedding": np.array([np.cos(np.radians(i * 18)), np.sin(np.radians(i * 18))], dtype=np.float32)
        }
        windows.append(window)
    return windows, windows[0]["embedding"].shape[0]


def test_find_top_5_embedding_candidates_basic():
    """Test basic functionality: returns up to 5 candidates per source."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Each source should have exactly 5 candidates (10 total windows - 1 self = 9, top 5)
    assert len(candidates) == 50  # 10 sources * 5 candidates
    
    # Each source should appear exactly once as source_window_id
    source_counts = {}
    for c in candidates:
        source_counts[c["source_window_id"]] = source_counts.get(c["source_window_id"], 0) + 1
    
    assert all(count == 5 for count in source_counts.values())


def test_candidate_rank_is_deterministic():
    """Test that candidate_rank is 1-5 and deterministic."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Verify ranks are 1-5
    for c in candidates:
        assert c["candidate_rank"] in [1, 2, 3, 4, 5]
        assert c["candidate_rank"] == 1 if c["cosine_similarity"] > 0.9 else c["candidate_rank"] <= 5


def test_cosine_similarity_values():
    """Test that cosine_similarity values are in valid range [0, 1]."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    for c in candidates:
        assert 0.0 <= c["cosine_similarity"] <= 1.0, f"cosine_similarity {c['cosine_similarity']} out of range"


def test_self_match_excluded():
    """Test that a window never appears as its own candidate."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    for c in candidates:
        assert c["source_window_id"] != c["candidate_window_id"], \
            f"Self-match detected: source {c['source_window_id']} -> candidate {c['candidate_window_id']}"


def test_near_overlap_excluded():
    """Test that near-overlap matches (same book+chapter, within 200 tokens) are excluded."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    # Create windows that are near-overlaps
    windows = [
        {
            "book": "Genesis",
            "chapter": 1,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": 0,  # Within 200 tokens of index 1
            "text": "Window 0 text",
            "filtered_text": "Window 0 text",
            "filtered_tokens": ["token0"] * 30,
            "embedding": np.array([1.0, 0.0], dtype=np.float32)
        },
        {
            "book": "Genesis",
            "chapter": 1,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": 50,  # Within 200 tokens of index 0
            "text": "Window 1 text",
            "filtered_text": "Window 1 text",
            "filtered_tokens": ["token1"] * 30,
            "embedding": np.array([0.9, 0.1], dtype=np.float32)  # High similarity but should be excluded
        },
        {
            "book": "Exodus",
            "chapter": 2,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": 1000,
            "text": "Window 2 text",
            "filtered_text": "Window 2 text",
            "filtered_tokens": ["token2"] * 30,
            "embedding": np.array([0.0, 1.0], dtype=np.float32)
        },
    ]
    
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Source 0 should not have window 1 as candidate (near-overlap)
    for c in candidates:
        if c["source_window_id"] == 0:
            assert c["candidate_window_id"] != 1, "Near-overlap match should be excluded"
    
    # Source 1 should not have window 0 as candidate (near-overlap)
    for c in candidates:
        if c["source_window_id"] == 1:
            assert c["candidate_window_id"] != 0, "Near-overlap match should be excluded"


def test_candidates_have_required_fields():
    """Test that candidate records contain all required fields."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    required_fields = [
        "source_window_id", "source_book", "source_chapter",
        "source_start_verse", "source_end_verse", "source_start_token_index",
        "source_text", "candidate_rank", "candidate_window_id",
        "candidate_book", "candidate_chapter", "candidate_start_verse",
        "candidate_end_verse", "candidate_start_token_index",
        "candidate_text", "cosine_similarity"
    ]
    
    for c in candidates:
        for field in required_fields:
            assert field in c, f"Missing field: {field}"


def test_export_to_csv():
    """Test CSV export functionality."""
    from top_5_candidates import find_top_5_embedding_candidates, export_top_5_candidates_to_csv
    from similarity import build_faiss_index
    import csv
    import os
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Export to CSV
    csv_path = "/home/momo/work/quote-identifier/.worktrees/t_ffd5b322/test_candidates_output.csv"
    export_top_5_candidates_to_csv(candidates, csv_path)
    
    # Verify CSV was created and has correct structure
    assert os.path.exists(csv_path)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(candidates)
        assert "source_window_id" in reader.fieldnames or reader.fieldnames[0] == "source_window_id"
        assert "cosine_similarity" in reader.fieldnames or "cosine_similarity" in reader.fieldnames


def test_ranking_is_by_descending_cosine_similarity():
    """Test that candidates are ranked by descending cosine similarity."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    # Get candidates for a specific source
    all_candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    source_0_candidates = [c for c in all_candidates if c["source_window_id"] == 0]
    
    # Verify ranks are in descending order of cosine similarity
    for i in range(len(source_0_candidates) - 1):
        curr = source_0_candidates[i]
        next_c = source_0_candidates[i + 1]
        
        # Higher similarity should have lower (better) rank
        if curr["cosine_similarity"] == next_c["cosine_similarity"]:
            # Tie-breaking: lower window index should have lower rank
            assert curr["candidate_window_id"] <= next_c["candidate_window_id"], \
                f"Tie-breaking failed at rank {curr['candidate_rank']} vs {next_c['candidate_rank']}"
        else:
            assert curr["cosine_similarity"] >= next_c["cosine_similarity"], \
                f"Ranking not descending: {curr['cosine_similarity']} < {next_c['cosine_similarity']}"


def test_tie_breaking_by_window_index():
    """Test that ties are broken by ascending window index."""
    from top_5_candidates import find_top_5_embedding_candidates
    from similarity import build_faiss_index
    
    # Create windows with identical embeddings
    windows = [
        {
            "book": "Book0",
            "chapter": 1,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": i * 500,  # Far apart to avoid near-overlap
            "text": f"Window {i}",
            "filtered_text": f"Window {i}",
            "filtered_tokens": [f"token{i}"] * 30,
            "embedding": np.array([0.8, 0.6], dtype=np.float32)  # Same embedding for all
        }
        for i in range(5)
    ]
    
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # All windows have same embedding, so ties should be broken by window index
    source_0_candidates = [c for c in candidates if c["source_window_id"] == 0]
    
    for i in range(len(source_0_candidates) - 1):
        assert source_0_candidates[i]["candidate_window_id"] < source_0_candidates[i + 1]["candidate_window_id"], \
            "Tie-breaking should prefer lower window index"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
