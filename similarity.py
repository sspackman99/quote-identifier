# similarity.py

from datasketch import MinHash, MinHashLSH
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import torch

# -------------------------------------------------
# 1. MINHASH
# -------------------------------------------------

def compute_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode("utf8"))
    return m


def add_minhashes(windows, text_field="filtered_text"):
    for window in windows:
        window["minhash"] = compute_minhash(window[text_field])
    return windows


def build_lsh(windows, threshold=0.6, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, window in enumerate(windows):
        lsh.insert(str(i), window["minhash"])
    return lsh


def find_minhash_matches(windows, lsh):
    matches = []
    for i, window in enumerate(windows):
        result = lsh.query(window["minhash"])
        for r in result:
            j = int(r)
            if j != i:
                matches.append((i, j, "minhash"))
    return matches

# -------------------------------------------------
# 2. EMBEDDINGS + FAISS (GPU ENABLED)
# -------------------------------------------------

def add_embeddings(windows,
                   text_field="filtered_text",
                   model_name="all-mpnet-base-v2",
                   batch_size=64):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer(model_name, device=device)
    
    texts = [w[text_field] for w in windows]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # IMPORTANT
    )
    
    for window, emb in zip(windows, embeddings):
        window["embedding"] = emb
    
    return windows


def build_faiss_index(windows):
    dim = len(windows[0]["embedding"])
    
    # Use inner product because embeddings are normalized
    index = faiss.IndexFlatIP(dim)
    
    embeddings = np.array([w["embedding"] for w in windows]).astype("float32")
    index.add(embeddings)
    
    return index, embeddings


def find_embedding_matches(windows, index, embeddings, k=5):
    matches = []
    for i, emb in enumerate(embeddings):
        D, I = index.search(np.array([emb]), k)
        for j in I[0]:
            if j != i:
                matches.append((i, j, "embedding"))
    return matches

# -------------------------------------------------
# 3. ADVANCED MULTI-SIGNAL SCORING
# -------------------------------------------------

def jaccard_similarity(m1, m2):
    return m1.jaccard(m2)


def detect_speaker_signal(window):
    """Very simple heuristic for speech detection."""
    speech_markers = {"say", "said", "saith", "spake", "cry", "command"}
    tokens = set(window["tokens"])
    return len(tokens & speech_markers) > 0


def score_match(window1, window2):
    # 1. Embedding similarity (normalized)
    emb_sim = float(np.dot(window1["embedding"], window2["embedding"]))
    
    # 2. MinHash similarity
    minhash_sim = jaccard_similarity(window1["minhash"], window2["minhash"])
    
    # 3. Vocabulary overlap
    vocab1 = set(window1["filtered_tokens"])
    vocab2 = set(window2["filtered_tokens"])
    vocab_overlap = (
        len(vocab1 & vocab2) / len(vocab1 | vocab2)
        if len(vocab1 | vocab2) > 0 else 0
    )
    
    # 4. Speaker signal boost
    speaker1 = detect_speaker_signal(window1)
    speaker2 = detect_speaker_signal(window2)
    speaker_boost = 0.05 if speaker1 != speaker2 else 0.0
    
    # 5. Combine signals
    final_score = (
        0.45 * emb_sim +
        0.30 * minhash_sim +
        0.20 * vocab_overlap +
        speaker_boost
    )
    
    return final_score


def filter_nearby_matches(windows, matches, min_token_distance=200):
    filtered = []
    for i, j, source in matches:
        w1 = windows[i]
        w2 = windows[j]
        
        # Skip if same chapter and too close
        if (w1["book"] == w2["book"] and w1["chapter"] == w2["chapter"]):
            distance = abs(w1["start_token_index"] - w2["start_token_index"])
            if distance < min_token_distance:
                continue
        
        filtered.append((i, j, source))
    
    return filtered


def get_top_matches(windows, matches, top_n=50):
    scored = []
    for i, j, source in matches:
        score = score_match(windows[i], windows[j])
        scored.append((score, i, j, source))
    
    scored.sort(reverse=True)
    return scored[:top_n]