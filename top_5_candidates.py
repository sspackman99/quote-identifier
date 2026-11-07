# top_5_candidates.py

import numpy as np

TOP_CANDIDATES = 5
MIN_TOKEN_DISTANCE = 200
MAX_SEARCH_K = 50


def find_top_5_embedding_candidates(windows, index, embeddings, top_k=TOP_CANDIDATES,
                                   query_ids=None, target_ids=None):
    """
    For each source window, find up to `top_k` candidates ranked by cosine similarity.

    Args:
        windows: List of window dicts with embeddings and metadata (book, chapter,
                start_token_index, text, etc.)
        index: FAISS index over normalized embeddings (IndexFlatIP)
        embeddings: NumPy array of embeddings, one row per entry in query_ids
                (shape: [len(query_ids), dim])
        top_k: How many candidates to keep per source window (default 5)
        query_ids: window ids to search from, in row order of `embeddings`
                (default: every window)
        target_ids: window id behind each row of `index` (default: every window,
                i.e. the index covers the full window list)

    Returns:
        List of candidate records, one per (source, candidate) pair.
        Each record contains: source_window_id, source_book, source_chapter,
        source_start_verse, source_end_verse, source_start_token_index, source_text,
        candidate_rank, candidate_window_id, candidate_book, candidate_chapter,
        candidate_start_verse, candidate_end_verse, candidate_start_token_index,
        candidate_text, cosine_similarity

    Ranking is by descending cosine similarity, ties broken by ascending
    candidate window id. Self-matches and near-overlap matches (same book and
    chapter, less than MIN_TOKEN_DISTANCE tokens apart) are excluded, and
    exclusion does not shorten the result when further valid candidates exist.

    `query_ids` / `target_ids` are what let one window list be searched against a
    SUBSET of windows — the cross-corpus case, where a Book of Mormon window must
    be answered from the Bible index alone.
    """
    candidates = []

    if len(windows) == 0 or len(embeddings) == 0:
        return candidates

    if query_ids is None:
        query_ids = range(len(windows))
    if target_ids is None:
        target_ids = range(len(windows))
    query_ids = list(query_ids)
    target_ids = list(target_ids)

    # Search wide enough that self-matches and near-overlaps can be dropped
    # without falling short of top_k.
    k = min(len(target_ids), MAX_SEARCH_K)

    for row, i in enumerate(query_ids):
        D, I = index.search(np.array([embeddings[row]]), k)

        # IndexFlatIP returns the inner product (embeddings are L2-normalized),
        # so no sign flip: D is the cosine similarity itself. Sort by
        # similarity descending, then by window id ascending for stable ties.
        ranked = sorted(
            ((float(sim), int(target_ids[j])) for sim, j in zip(D[0], I[0])),
            key=lambda pair: (-pair[0], pair[1]),
        )

        w1 = windows[i]
        rank = 0

        for cos_sim, j in ranked:
            if j == i:  # Skip self-match
                continue

            w2 = windows[j]

            # Skip near-overlap matches
            if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
                distance = abs(w1["start_token_index"] - w2["start_token_index"])
                if distance < MIN_TOKEN_DISTANCE:
                    continue

            rank += 1
            candidates.append({
                "source_window_id": i,
                "source_book": w1["book"],
                "source_chapter": w1["chapter"],
                "source_start_verse": w1["start_verse"],
                "source_end_verse": w1["end_verse"],
                "source_start_token_index": w1["start_token_index"],
                "source_text": w1["text"],
                "candidate_rank": rank,
                "candidate_window_id": j,
                "candidate_book": w2["book"],
                "candidate_chapter": w2["chapter"],
                "candidate_start_verse": w2["start_verse"],
                "candidate_end_verse": w2["end_verse"],
                "candidate_start_token_index": w2["start_token_index"],
                "candidate_text": w2["text"],
                "cosine_similarity": cos_sim,
            })

            if rank == top_k:
                break

    return candidates


def find_candidates_between(windows, embeddings, query_ids, target_ids, top_k=TOP_CANDIDATES):
    """Top-k candidates for each query window, searched ONLY among target windows.

    This is the cross-corpus search. Pooling two corpora into one index and
    filtering afterwards is not equivalent: a Book of Mormon window's nearest
    neighbours are usually other Book of Mormon windows (the same passage quoted
    elsewhere, or a parallel account), so the Bible chapter it actually quotes
    can sit far outside its top-k. Searching the target corpus alone makes the
    question "which Bible passage is closest to this verse" instead of "which of
    70,000 windows is closest".
    """
    import faiss

    embeddings = np.asarray(embeddings, dtype="float32")
    target_positions = np.asarray(list(target_ids), dtype="int64")
    query_positions = np.asarray(list(query_ids), dtype="int64")

    if len(target_positions) == 0 or len(query_positions) == 0:
        return []

    subset = embeddings[target_positions]
    index = faiss.IndexFlatIP(subset.shape[1])
    index.add(subset)

    return find_top_5_embedding_candidates(
        windows, index, embeddings[query_positions], top_k=top_k,
        query_ids=query_positions, target_ids=target_positions,
    )


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

    df = pd.DataFrame(candidates, columns=fieldnames)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(candidates)} candidate records to {filepath}")
