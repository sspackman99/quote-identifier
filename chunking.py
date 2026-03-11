
# -------------------------------------------------
# 7. Build Overlapping Windows Per Chapter
# -------------------------------------------------

from collections import defaultdict


def build_chapter_token_stream(processed_verses):
    """
    Groups verses by (book, chapter) and creates a continuous
    token stream per chapter while preserving verse mapping.
    """
    
    chapters = defaultdict(list)
    
    # Group verses
    for verse in processed_verses:
        key = (verse["book"], verse["chapter"])
        chapters[key].append(verse)
    
    chapter_streams = []
    
    for (book, chapter), verses in chapters.items():
        
        all_tokens = []
        token_to_verse = []
        
        for verse in verses:
            for token in verse["tokens"]:
                all_tokens.append(token)
                token_to_verse.append(verse["verse"])
        
        chapter_streams.append({
            "corpus": "bom",
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
    """
    
    step = window_size - overlap
    windows = []
    
    for chapter_data in chapter_streams:
        
        tokens = chapter_data["tokens"]
        token_to_verse = chapter_data["token_to_verse"]
        
        for i in range(0, len(tokens) - window_size + 1, step):
            
            window_tokens = tokens[i:i+window_size]
            verse_span = token_to_verse[i:i+window_size]
            
            start_verse = verse_span[0]
            end_verse = verse_span[-1]
            
            windows.append({
                "corpus": chapter_data["corpus"],
                "book": chapter_data["book"],
                "chapter": chapter_data["chapter"],
                "start_token_index": i,
                "end_token_index": i + window_size,
                "start_verse": start_verse,
                "end_verse": end_verse,
                "tokens": window_tokens,
                "text": " ".join(window_tokens)
            })
    
    return windows