# Top-5 Cosine Similarity Candidate Retrieval Specification

## Summary

This specification defines the exact behavior for retrieving exactly five candidate scripture windows per source window based on cosine similarity.

---

## Current Behavior Analysis

### 1. FAISS k=5 Meaning (similarity.py:86-93)

```python
def find_embedding_matches(windows, index, embeddings, k=5):
    matches = []
    for i, emb in enumerate(embeddings):
        D, I = index.search(np.array([emb]), k)
        for j in I[0]:
            if j != i:
                matches.append((i, j, "embedding"))
    return matches
```

**What k=5 means:** For each source window embedding, FAISS returns the 5 nearest neighbors (highest inner product) from the **global index**. This is **NOT per-source retrieval** — it's a global nearest-neighbor search where each query returns its own top-5.

**Key properties:**
- Each source window gets up to 5 candidates (fewer if corpus is small)
- Candidates are the globally nearest embeddings, not filtered by any prior stage
- Self-matches are excluded with `if j != i`

### 2. No Subsequent Collapsing or Global Reranking

The current pipeline:
1. Find MinHash matches (LSH threshold=0.6)
2. Find embedding matches (FAISS k=5 per source)
3. Concatenate: `all_matches = minhash_matches + embedding_matches`
4. Score all matches with `score_match()` (composite: 45% embedding, 30% MinHash, 20% vocab)
5. Return top-N (default 50) by composite score

**Conclusion:** FAISS k=5 candidates are NOT collapsed or globally reranked. They remain as a separate match source alongside MinHash matches, and later compete in the composite scoring stage.

### 3. MinHash Does Not Affect FAISS Selection

MinHash operates independently:
- `find_minhash_matches()` uses LSH to find approximate nearest neighbors
- Results are tuples `(i, j, "minhash")`
- No influence on which FAISS neighbors are returned

**Conclusion:** MinHash/composite scoring does NOT affect FAISS candidate selection. The two pipelines are orthogonal.

### 4. Self-Matches and Nearby Overlapping Matches

**Self-matches:** Excluded via `if j != i` in `find_embedding_matches()`.

**Nearby overlapping matches:** Handled by `filter_nearby_matches()` (similarity.py:141-155):
```python
def filter_nearby_matches(windows, matches, min_token_distance=200):
    for i, j, source in matches:
        w1 = windows[i]
        w2 = windows[j]
        if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
            distance = abs(w1["start_token_index"] - w2["start_token_index"])
            if distance < min_token_distance:
                continue
```

**Note:** This filtering happens AFTER FAISS retrieval, so it can remove some of the top-5 candidates.

### 5. Metadata Preservation

Current metadata in matches:
- `i, j` — source and target window indices
- `source` — match method ("embedding" or "minhash")
- Composite score from `score_match()`

**Missing for candidate retrieval:**
- Cosine similarity value (currently only present in composite score)
- Raw embedding similarity
- Source window metadata (book, chapter, verses, tokens)
- Target window metadata

---

## Desired Behavior Specification

### Objective

For each source scripture window, identify exactly five candidate scripture windows based on cosine similarity, preserving the source-window → ranked-candidate relationship.

### Candidate Selection Rules

1. **Per-source retrieval:** Each source window independently retrieves its own top-5 candidates
2. **Cosine similarity:** Use raw embedding inner product (embeddings are normalized)
3. **Self-match exclusion:** Exclude the source window itself (`j != i`)
4. **No MinHash influence:** FAISS selection is independent of MinHash pipeline
5. **Deterministic ordering:** Sort by cosine similarity descending; break ties by target window index ascending

### Exclusion Rules

1. **Self-match exclusion:** Already handled by `if j != i`
2. **Near-overlap exclusion:** Apply `filter_nearby_matches()` with `min_token_distance=200` (same book+chapter, within 200 tokens)

**Important:** Near-overlap filtering occurs AFTER candidate selection, meaning it may reduce the final candidate count below 5.

### Output Format

Each candidate record must include:

| Field | Type | Description |
|-------|------|-------------|
| `source_window_id` | int | Index of source window |
| `source_book` | str | Bible book name |
| `source_chapter` | int | Chapter number |
| `source_start_verse` | int | Start verse |
| `source_end_verse` | int | End verse |
| `source_start_token_index` | int | Token index in BOM |
| `source_text` | str | Full window text |
| `candidate_rank` | int | Rank 1..5 (or fewer if candidates exhausted) |
| `candidate_window_id` | int | Index of candidate window |
| `candidate_book` | str | Bible book name |
| `candidate_chapter` | int | Chapter number |
| `candidate_start_verse` | int | Start verse |
| `candidate_end_verse` | int | End verse |
| `candidate_start_token_index` | int | Token index in BOM |
| `candidate_text` | str | Full window text |
| `cosine_similarity` | float | Raw embedding inner product (0.0 to 1.0) |

### Tie-Breaking Behavior

When two candidates have identical cosine similarity:
1. Prefer lower `candidate_window_id` (ascending)
2. This ensures deterministic output across runs

### Global Top-N vs Per-Source Top-5

**This is per-source top-5, NOT global top-N.**

- Each source window independently finds its own 5 best matches
- A candidate can appear as a match for multiple sources
- Total candidates = min(5, N) × M (where M = number of sources)

**Contrast with current behavior:**
- Current: FAISS k=5 per source (same as desired) ✓
- Current: Global top-N selection after scoring (different) ✗
- Desired: Per-source top-5 before composite scoring ✓

---

## Implementation Targets

### Functions to Modify/Add

#### 1. `find_top_5_embedding_candidates(windows, index, embeddings)`

**Purpose:** Extract top-5 candidates per source window with cosine similarity.

**Input:**
- `windows`: List of window dicts with embeddings
- `index`: FAISS index
- `embeddings`: NumPy array of embeddings

**Output:** List of dicts, one per (source, candidate) pair

```python
def find_top_5_embedding_candidates(windows, index, embeddings):
    """
    For each source window, find up to 5 candidates ranked by cosine similarity.
    
    Returns list of candidate records with full metadata and cosine similarity.
    """
    candidates = []
    
    for i, emb in enumerate(embeddings):
        D, I = index.search(np.array([emb]), k=min(5, len(windows)))
        
        for rank, j in enumerate(I[0]):
            if j == i:  # Skip self-match
                continue
            
            # Apply near-overlap filtering
            w1, w2 = windows[i], windows[j]
            if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
                if abs(w1["start_token_index"] - w2["start_token_index"]) < 200:
                    continue
            
            # Build candidate record
            cos_sim = float(D[0][j])  # Cosine similarity (inner product)
            
            candidates.append({
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
            })
    
    return candidates
```

#### 2. CSV Output Format

```
source_window_id,source_book,source_chapter,source_start_verse,source_end_verse,source_start_token_index,source_text,candidate_rank,candidate_window_id,candidate_book,candidate_chapter,candidate_start_verse,candidate_end_verse,candidate_start_token_index,candidate_text,cosine_similarity
0,Genesis,1,1,3,0,"In the beginning God created...",1,2,Genesis,1,3,45,"And the earth was without form...",0.9234
```

#### 3. Integration with Existing Pipeline

**Option A: Separate output file**
- Run `find_top_5_embedding_candidates()` independently
- Output to `top_5_candidates.csv`
- No modification to existing `get_top_matches()` flow

**Option B: Enrich existing matches**
- Modify `find_embedding_matches()` to return enriched records
- Pass through to `get_top_matches()` for composite scoring
- Higher priority for embedding matches with high cosine similarity

**Recommendation:** Option A for clarity and independence.

---

## Acceptance Criteria

1. **Per-source top-5:** Each source window has up to 5 candidates based on cosine similarity
2. **Cosine similarity values:** Raw inner product values (0.0 to 1.0) included in output
3. **Self-match exclusion:** Source window never appears as its own candidate
4. **Near-overlap filtering:** Candidates within 200 tokens in same book+chapter are excluded
5. **Tie-breaking:** Deterministic ordering via ascending window index
6. **No MinHash influence:** FAISS selection independent of MinHash pipeline
7. **Metadata preservation:** Full source and candidate metadata in output
8. **Separate from composite scoring:** Top-5 retrieval distinct from composite scoring pipeline

---

## Notes

### Why Per-Source Top-5, Not Global Top-N?

The downstream LLM evaluation evaluates each (source, candidate) pair independently. Per-source top-5 ensures:
- Each source has comparable candidate pool size
- No source is disadvantaged by having fewer matches
- Consistent evaluation across the corpus

### Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| Per-source top-5 | Consistent evaluation, fair comparison | May include lower-quality candidates |
| Global top-N | Highest-quality matches overall | Uneven candidate pools, some sources underrepresented |

**Decision:** Per-source top-5 chosen for evaluation consistency.
