import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from corpora import load_corpus_text, parse_bom_text

# Optional lemmatization (recommended)
USE_LEMMATIZATION = True

# Defaults so tokenize() cannot hit NameError when lemmatization is off or
# nltk is missing.
lemmatizer = None
nltk_available = False

if USE_LEMMATIZATION:
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer
        
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")
            nltk.download("omw-1.4")
        
        lemmatizer = WordNetLemmatizer()
        
        nltk_available = True
    except ImportError:
        # NLTK not available - lemmatization will be skipped
        lemmatizer = None
        nltk_available = False



# -------------------------------------------------
# 1. Load Text
# -------------------------------------------------

def load_text(path: str) -> str:
    # utf-8-sig strips the BOM that the Gutenberg text files carry
    return Path(path).read_text(encoding="utf-8-sig")


# Corpus text is NOT stored in this repository (see data/README.md): it is a
# separate download. These are the places main.py looks when no path is given.
DEFAULT_CORPUS_CANDIDATES = ("data/bom.txt", "bom.txt")
CORPUS_ENV_VAR = "QI_CORPUS"


def resolve_corpus_path(path: Optional[str] = None) -> Path:
    """Locate the corpus text file, or explain exactly what is missing.

    Resolution order: explicit path, $QI_CORPUS, then DEFAULT_CORPUS_CANDIDATES.
    Raises FileNotFoundError whose message names every place that was checked.
    """
    if path:
        candidate = Path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"corpus file not found: {candidate}. Pass an existing path with "
            f"--corpus, or set {CORPUS_ENV_VAR}."
        )

    from_env = os.environ.get(CORPUS_ENV_VAR)
    if from_env:
        candidate = Path(from_env)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"{CORPUS_ENV_VAR} points at a file that does not exist: {candidate}"
        )

    checked = []
    for candidate_str in DEFAULT_CORPUS_CANDIDATES:
        candidate = Path(candidate_str)
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "no corpus text found. This repository does not ship the scripture text. "
        f"Looked for: {', '.join(checked)}. Put the file at data/bom.txt (see "
        f"data/README.md), pass --corpus PATH, or set {CORPUS_ENV_VAR}."
    )


# -------------------------------------------------
# 2. Text Normalization
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
# 3. Tokenization
# -------------------------------------------------

def tokenize(text: str) -> List[str]:
    tokens = text.split()
    
    if USE_LEMMATIZATION and nltk_available:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


# -------------------------------------------------
# 4. Full Preprocessing Pipeline
# -------------------------------------------------

def preprocess_corpus(corpus_id: str,
                      path: str,
                      remove_punctuation: bool = False) -> List[Dict]:
    """Parse a corpus through its own parser, then normalize and tokenize.

    The parser (see corpora.py) returns canonical verse records — corpus,
    book_id, book, chapter, verse, text, source_ref, source_line — and this step
    adds clean_text and tokens without changing any of those fields.
    """
    raw_text = load_text(path)
    parsed_verses = load_corpus_text(corpus_id, raw_text)

    processed_verses = []

    for parsed in parsed_verses:
        clean_text = normalize_text(
            parsed["text"],
            lowercase=True,
            remove_punctuation=remove_punctuation,
        )

        record = dict(parsed)
        record["original_text"] = parsed["text"]
        record["clean_text"] = clean_text
        record["tokens"] = tokenize(clean_text)
        processed_verses.append(record)

    return processed_verses


def preprocess_bom(path: str,
                   remove_punctuation: bool = False) -> List[Dict]:
    """Book of Mormon entry point, kept for existing callers.

    Goes through the shared corpus registry like every other source, so the
    output now carries the canonical fields (book_id, integer chapter/verse,
    source_ref, source_line).
    """
    return preprocess_corpus("bom", path, remove_punctuation=remove_punctuation)


def parse_scripture_text(text: str) -> List[Dict]:
    """Compatibility alias for the Book of Mormon parser in corpora.py."""
    return parse_bom_text(text)
