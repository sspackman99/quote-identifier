
# -------------------------------------------------
# 7. Build Overlapping Windows Per Chapter
# -------------------------------------------------

from collections import defaultdict

from corpora import slugify


def build_chapter_token_stream(processed_verses):
    """
    Groups verses by (corpus, book, chapter) and creates a continuous
    token stream per chapter while preserving verse mapping.

    Canonical verse records carry the corpus and a stable book_id, so two
    corpora with the same book/chapter never collapse into one chapter stream.
    """
    
    chapters = defaultdict(list)
    
    # Group verses
    for verse in processed_verses:
        key = (
            verse.get("corpus", "bom"),
            verse.get("book_id") or slugify(verse["book"]),
            verse["book"],
            verse["chapter"],
        )
        chapters[key].append(verse)
    
    chapter_streams = []
    
    for (corpus_id, book_id, book, chapter), verses in chapters.items():
        
        all_tokens = []
        token_to_verse = []
        
        for verse in verses:
            for token in verse["tokens"]:
                all_tokens.append(token)
                token_to_verse.append(verse["verse"])
        
        chapter_streams.append({
            "corpus": corpus_id,
            "book_id": book_id,
            "book": book,
            "chapter": chapter,
            "tokens": all_tokens,
            "token_to_verse": token_to_verse
        })
    
    return chapter_streams


def create_overlapping_windows(chapter_streams,
                               window_size=30,
                               overlap=15):
    """
    Creates overlapping token windows per chapter.

    Windows never cross a chapter boundary, and every token in a chapter is
    covered by at least one window: when the last step boundary would leave a
    tail shorter than the window, one extra window anchored to the chapter end
    is appended (unless that start was already emitted). A chapter shorter than
    the window produces a single window holding the whole chapter.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size")

    step = window_size - overlap
    windows = []

    for chapter_data in chapter_streams:

        tokens = chapter_data["tokens"]
        token_to_verse = chapter_data["token_to_verse"]
        total = len(tokens)

        if total == 0:
            continue

        if total <= window_size:
            starts = [0]
        else:
            starts = list(range(0, total - window_size + 1, step))
            final_start = total - window_size
            if starts[-1] != final_start:
                starts.append(final_start)

        for i in starts:

            end = min(i + window_size, total)
            verse_span = token_to_verse[i:end]

            windows.append({
                "corpus": chapter_data["corpus"],
                "book_id": chapter_data.get("book_id"),
                "book": chapter_data["book"],
                "chapter": chapter_data["chapter"],
                "start_token_index": i,
                "end_token_index": end,
                "start_verse": verse_span[0],
                "end_verse": verse_span[-1],
                "tokens": tokens[i:end],
                "text": " ".join(tokens[i:end])
            })
    
    return windows