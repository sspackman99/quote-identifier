import re
from pathlib import Path
from typing import List, Dict, Optional

# Optional lemmatization (recommended)
USE_LEMMATIZATION = True

if USE_LEMMATIZATION:
    import nltk
    from nltk.stem import WordNetLemmatizer
    
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    
    lemmatizer = WordNetLemmatizer()


# -------------------------------------------------
# 1. Load Text
# -------------------------------------------------

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


# -------------------------------------------------
# 2. Split into Book + Chapter Sections
# Example header:
# "1 Nephi Chapter 1"
# -------------------------------------------------

def split_book_chapters(text: str):
    pattern = r'(\d+\s+[A-Za-z]+\s+Chapter\s+\d+)'
    parts = re.split(pattern, text)
    
    sections = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i+1].strip()
        sections.append((header, body))
    
    return sections


def parse_header(header: str):
    # Example: "1 Nephi Chapter 1"
    match = re.match(r'(\d+)\s+([A-Za-z]+)\s+Chapter\s+(\d+)', header)
    if match:
        book_number = match.group(1)
        book_name = match.group(2)
        chapter_number = int(match.group(3))
        full_book = f"{book_number} {book_name}"
        return full_book, chapter_number
    return None, None


# -------------------------------------------------
# 3. Split into Verses
# Example format:
# "1:1 I, Nephi..."
# -------------------------------------------------

def split_verses(chapter_body: str):
    verse_pattern = r'(\d+:\d+)\s'
    parts = re.split(verse_pattern, chapter_body)
    
    verses = []
    for i in range(1, len(parts), 2):
        verse_id = parts[i]
        verse_text = parts[i+1].strip()
        verses.append((verse_id, verse_text))
    
    return verses


# -------------------------------------------------
# 4. Text Normalization
# -------------------------------------------------

def normalize_text(text: str,
                   lowercase: bool = True,
                   remove_punctuation: bool = False) -> str:
    
    if lowercase:
        text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    return text.strip()


# -------------------------------------------------
# 5. Tokenization
# -------------------------------------------------

def tokenize(text: str) -> List[str]:
    tokens = text.split()
    
    if USE_LEMMATIZATION:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


# -------------------------------------------------
# 6. Full Preprocessing Pipeline
# -------------------------------------------------

def preprocess_bom(path: str,
                   remove_punctuation: bool = False) -> List[Dict]:
    
    raw_text = load_text(path)
    sections = split_book_chapters(raw_text)
    
    processed_verses = []
    
    for header, body in sections:
        book, chapter = parse_header(header)
        verses = split_verses(body)
        
        for verse_id, verse_text in verses:
            
            clean_text = normalize_text(
                verse_text,
                lowercase=True,
                remove_punctuation=remove_punctuation
            )
            
            tokens = tokenize(clean_text)
            
            processed_verses.append({
                "corpus": "bom",
                "book": book,
                "chapter": chapter,
                "verse": verse_id,
                "original_text": verse_text,
                "clean_text": clean_text,
                "tokens": tokens
            })
    
    return processed_verses

