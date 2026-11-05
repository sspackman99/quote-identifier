import os
import re
from pathlib import Path
from typing import Dict, List, Optional

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
    
    if USE_LEMMATIZATION and nltk_available:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


# -------------------------------------------------
# 6. Structure-aware scripture parsing
# -------------------------------------------------
#
# split_book_chapters/parse_header/split_verses above understand exactly one
# shape: a NUMBERED book with an explicit "N Book Chapter N" header. A real
# corpus (the Project Gutenberg Book of Mormon) also contains unnumbered books
# ("Mosiah Chapter 1"), books with no chapter heading at all (Enos, Jarom,
# Omni), and an ordinal title whose verses are marked inline ("FOURTH NEPHI"
# followed by "4 Nephi 1:1 ..."), plus a table of contents and licence front
# matter.
#
# Fed through the narrow splitter, those books do not raise an error — their
# text is swallowed into the body of the previous numbered chapter, so verses
# come back attributed to the wrong book and chapter. parse_scripture_text()
# scans line by line and tracks book/chapter state instead, so each verse keeps
# the reference it actually came from.

_ORDINALS = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4"}

_BOOK_TITLES = {
    "NEPHI": "Nephi",
    "JACOB": "Jacob",
    "ENOS": "Enos",
    "JAROM": "Jarom",
    "OMNI": "Omni",
    "MOSIAH": "Mosiah",
    "ALMA": "Alma",
    "HELAMAN": "Helaman",
    "MORMON": "Mormon",
    "ETHER": "Ether",
    "MORONI": "Moroni",
}

# " THE BOOK OF MOSIAH", " THE SECOND BOOK OF NEPHI",
# " THE FIRST BOOK OF NEPHI HIS REIGN AND MINISTRY (1 Nephi)", " FOURTH NEPHI"
_TITLE_RE = re.compile(
    r'^\s*(?:THE\s+)?(?:(?P<ordinal>FIRST|SECOND|THIRD|FOURTH)\s+)?'
    r'(?:BOOK\s+OF\s+)?(?P<title>[A-Z]+(?:\s+[A-Z]+)*)'
    r'\s*(?:\((?P<abbr>[^)]+)\))?\s*$'
)

# "Mosiah Chapter 12", "1 Nephi Chapter 1"
_CHAPTER_HEADER_RE = re.compile(
    r'^(?:(?P<number>[1-4])\s+)?(?P<book>[A-Z][A-Za-z]+)\s+Chapter\s+(?P<chapter>\d+)\s*$'
)

# "1:1 I, Nephi, having been born..."
_VERSE_RE = re.compile(r'^(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.*)$')

# "4 Nephi 1:1 And it came to pass..."
_INLINE_VERSE_RE = re.compile(
    r'^(?P<number>[1-4])\s+(?P<book>[A-Z][A-Za-z]+)\s+'
    r'(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.*)$'
)


def _canonical_book(title: str,
                    ordinal: Optional[str],
                    abbr: Optional[str]) -> Optional[str]:
    """Map a title line's parts onto the book name used in verse records."""
    if abbr:
        # "THE FIRST BOOK OF NEPHI HIS REIGN AND MINISTRY (1 Nephi)"
        return abbr.strip()

    base = _BOOK_TITLES.get(title.strip().upper())
    if base is None:
        return None

    return f"{_ORDINALS[ordinal]} {base}" if ordinal else base


def _is_title_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return "BOOK OF" in stripped.upper() or (
        stripped.upper() == stripped and any(ch.isalpha() for ch in stripped)
    )


def parse_scripture_text(text: str) -> List[Dict]:
    """Parse scripture text into verse records with correct book/chapter refs.

    Handles numbered books with chapter headers, unnumbered books, books whose
    only chapter has no heading, and inline "4 Nephi 1:1" verse markers. Front
    matter before the first chapter heading (licence, table of contents) is
    skipped, as is any text before a book has been established.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start = None
    for index, line in enumerate(lines):
        if _CHAPTER_HEADER_RE.match(line.strip()):
            start = index
            break
        title_match = _TITLE_RE.match(line) if _is_title_line(line) else None
        if title_match and _canonical_book(
            title_match.group("title"),
            title_match.group("ordinal"),
            title_match.group("abbr"),
        ):
            start = index
            break
    if start is None:
        return []

    verses: List[Dict] = []
    book: Optional[str] = None
    chapter: Optional[int] = None
    current: Optional[Dict] = None

    def close_current():
        nonlocal current
        if current is not None:
            current["text"] = re.sub(r'\s+', ' ', current["text"]).strip()
            verses.append(current)
            current = None

    for raw_line in lines[start:]:
        line = raw_line.strip()

        if _is_title_line(raw_line):
            title_match = _TITLE_RE.match(raw_line)
            if title_match:
                candidate = _canonical_book(
                    title_match.group("title"),
                    title_match.group("ordinal"),
                    title_match.group("abbr"),
                )
                if candidate:
                    close_current()
                    book = candidate
                    # Single-chapter books have no chapter heading of their own.
                    chapter = 1
                    continue

        header_match = _CHAPTER_HEADER_RE.match(line)
        if header_match:
            close_current()
            number = header_match.group("number")
            name = header_match.group("book")
            book = f"{number} {name}" if number else name
            chapter = int(header_match.group("chapter"))
            continue

        inline_match = _INLINE_VERSE_RE.match(line)
        if inline_match:
            close_current()
            book = f"{inline_match.group('number')} {inline_match.group('book')}"
            chapter = int(inline_match.group("chapter"))
            current = {
                "book": book,
                "chapter": chapter,
                "verse": f"{inline_match.group('chapter')}:{inline_match.group('verse')}",
                "text": inline_match.group("text"),
            }
            continue

        verse_match = _VERSE_RE.match(line)
        if verse_match and book is not None and chapter is not None:
            close_current()
            current = {
                "book": book,
                "chapter": chapter,
                "verse": f"{verse_match.group('chapter')}:{verse_match.group('verse')}",
                "text": verse_match.group("text"),
            }
            continue

        if current is not None and line:
            current["text"] += " " + line

    close_current()
    return verses


# -------------------------------------------------
# 7. Full Preprocessing Pipeline
# -------------------------------------------------

def preprocess_bom(path: str,
                   remove_punctuation: bool = False) -> List[Dict]:
    """Load a scripture text file and return normalized, tokenized verses."""
    raw_text = load_text(path)
    parsed_verses = parse_scripture_text(raw_text)

    processed_verses = []

    for parsed in parsed_verses:
        verse_text = parsed["text"]

        clean_text = normalize_text(
            verse_text,
            lowercase=True,
            remove_punctuation=remove_punctuation
        )

        tokens = tokenize(clean_text)

        processed_verses.append({
            "corpus": "bom",
            "book": parsed["book"],
            "chapter": parsed["chapter"],
            "verse": parsed["verse"],
            "original_text": verse_text,
            "clean_text": clean_text,
            "tokens": tokens
        })

    return processed_verses
