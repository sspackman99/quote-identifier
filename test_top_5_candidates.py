# test_top_5_candidates.py

import pytest
import numpy as np


def build_faiss_index(windows):
    """Build a FAISS inner-product index over pre-computed window embeddings.

    Kept local to the tests on purpose: similarity.py imports
    sentence_transformers/torch/sklearn/datasketch at module level, so importing
    it here would make this suite require the full ML stack even though the
    tests only supply numpy embeddings and need faiss.
    """
    import faiss

    dim = len(windows[0]["embedding"])
    index = faiss.IndexFlatIP(dim)
    embeddings = np.array([w["embedding"] for w in windows]).astype("float32")
    index.add(embeddings)
    return index, embeddings


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
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Verify ranks are exactly 1..5 for every source, and stable across runs
    by_source = {}
    for c in candidates:
        by_source.setdefault(c["source_window_id"], []).append(c)

    assert len(by_source) == 10, "every source window must produce candidates"
    for source_id, records in by_source.items():
        assert [r["candidate_rank"] for r in records] == [1, 2, 3, 4, 5], \
            f"source {source_id} ranks are {[r['candidate_rank'] for r in records]}"

    again = find_top_5_embedding_candidates(windows, index, embeddings)
    assert again == candidates, "candidate output must be deterministic"


def test_cosine_similarity_is_true_inner_product():
    """cosine_similarity must equal the normalized dot product, never negated."""
    from top_5_candidates import find_top_5_embedding_candidates

    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)

    candidates = find_top_5_embedding_candidates(windows, index, embeddings)

    for c in candidates:
        expected = float(np.dot(embeddings[c["source_window_id"]],
                                embeddings[c["candidate_window_id"]]))
        assert c["cosine_similarity"] == pytest.approx(expected, abs=1e-5), \
            f"source {c['source_window_id']} -> candidate {c['candidate_window_id']}"


def test_near_overlap_exclusion_does_not_reduce_candidate_count():
    """Dropping a near-overlap must be backfilled so the source still gets 5."""
    from top_5_candidates import find_top_5_embedding_candidates

    def window(book, chapter, token_index, embedding):
        return {
            "book": book,
            "chapter": chapter,
            "start_verse": 1,
            "end_verse": 3,
            "start_token_index": token_index,
            "text": f"{book} {chapter} @{token_index}",
            "filtered_text": f"{book} {chapter} @{token_index}",
            "filtered_tokens": ["token"] * 30,
            "embedding": np.array(embedding, dtype=np.float32),
        }

    # Source 0 and window 1 are a near-overlap pair (same book+chapter, 50 tokens apart)
    # and window 1 is the most similar candidate, so it must be skipped AND replaced.
    windows = [
        window("Genesis", 1, 0, [1.0, 0.0]),
        window("Genesis", 1, 50, [1.0, 0.0]),
    ]
    for n in range(1, 7):
        magnitude = (1.0 - n * 0.1) ** 0.5
        windows.append(window(f"Book{n}", 1, 10_000 * n, [magnitude, (1.0 - magnitude ** 2) ** 0.5]))

    index, embeddings = build_faiss_index(windows)
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)

    source_0 = [c for c in candidates if c["source_window_id"] == 0]
    assert len(source_0) == 5, f"expected 5 backfilled candidates, got {len(source_0)}"
    assert 1 not in [c["candidate_window_id"] for c in source_0], "near-overlap was not excluded"
    assert [c["candidate_rank"] for c in source_0] == [1, 2, 3, 4, 5]


def test_cosine_similarity_values():
    """Test that cosine_similarity values are in valid range [0, 1]."""
    from top_5_candidates import find_top_5_embedding_candidates
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    for c in candidates:
        assert 0.0 <= c["cosine_similarity"] <= 1.0, f"cosine_similarity {c['cosine_similarity']} out of range"


def test_self_match_excluded():
    """Test that a window never appears as its own candidate."""
    from top_5_candidates import find_top_5_embedding_candidates
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    for c in candidates:
        assert c["source_window_id"] != c["candidate_window_id"], \
            f"Self-match detected: source {c['source_window_id']} -> candidate {c['candidate_window_id']}"


def test_near_overlap_excluded():
    """Test that near-overlap matches (same book+chapter, within 200 tokens) are excluded."""
    from top_5_candidates import find_top_5_embedding_candidates
    
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


def test_export_to_csv(tmp_path):
    """Test CSV export functionality."""
    from top_5_candidates import find_top_5_embedding_candidates, export_top_5_candidates_to_csv
    import csv
    import os
    
    windows, emb_dim = create_test_windows()
    index, embeddings = build_faiss_index(windows)
    
    candidates = find_top_5_embedding_candidates(windows, index, embeddings)
    
    # Export to CSV
    csv_path = str(tmp_path / "test_candidates_output.csv")

    export_top_5_candidates_to_csv(candidates, csv_path)
    
    # Verify CSV was created and has correct structure
    assert os.path.exists(csv_path)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == len(candidates)
        assert reader.fieldnames[0] == "source_window_id"
        assert reader.fieldnames[-1] == "cosine_similarity"
        float(rows[0]["cosine_similarity"])  # parses as a number



def test_ranking_is_by_descending_cosine_similarity():
    """Test that candidates are ranked by descending cosine similarity."""
    from top_5_candidates import find_top_5_embedding_candidates
    
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
