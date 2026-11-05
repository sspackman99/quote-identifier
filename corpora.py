"""Canonical corpus model, book registry, and per-source parsers.

Every parser returns the SAME record shape, whatever the source looks like:

    corpus         corpus id, e.g. "bom" or "kjv"
    book_id        stable identifier, e.g. "isaiah" (identical from any source)
    book           display name as the corpus standardises it, e.g. "Isaiah"
    chapter        integer chapter number
    verse          integer verse number
    text           verse text exactly as the source gives it
    source_ref     reference string as printed in the source (audit trail)
    source_line    1-based line number in the source file (audit trail)

Parsers are source-specific by design: the Book of Mormon text is a Gutenberg
document with book titles and chapter headings, the King James text is one
tab-separated line per verse. Each returns canonical records; nothing downstream
needs to know which one produced them.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CANONICAL_FIELDS = (
    "corpus", "book_id", "book", "chapter", "verse",
    "text", "source_ref", "source_line",
)


class CorpusFormatError(ValueError):
    """A source file did not match the layout its parser expects."""


# -------------------------------------------------
# Book registry
# -------------------------------------------------

BOM_BOOKS = [
    "1 Nephi", "2 Nephi", "Jacob", "Enos", "Jarom", "Omni", "Words of Mormon",
    "Mosiah", "Alma", "Helaman", "3 Nephi", "4 Nephi", "Mormon", "Ether",
    "Moroni",
]

KJV_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua",
    "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job",
    "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah",
    "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai",
    "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James",
    "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
]

# Variant spellings that appear in the wild, mapped to the canonical name.
BOOK_ALIASES = {
    "Psalm": "Psalms",
    "Psalter": "Psalms",
    "Song of Songs": "Song of Solomon",
    "Canticles": "Song of Solomon",
    "Revelations": "Revelation",
    "The Revelation": "Revelation",
    "Canticle of Canticles": "Song of Solomon",
    "1 Nephi his reign and ministry": "1 Nephi",
}

# name -> (corpus ids that contain it)
_BOOK_INDEX: Dict[str, Dict[str, str]] = {}


def slugify(name: str) -> str:
    """Stable identifier for a book name: '1 Nephi' -> '1-nephi'."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _register(corpus_id: str, names: List[str]) -> None:
    for name in names:
        _BOOK_INDEX.setdefault(name, {})[corpus_id] = slugify(name)


_register("bom", BOM_BOOKS)
_register("kjv", KJV_BOOKS)


def books_for(corpus_id: str) -> List[str]:
    return sorted(
        (name for name, corpora in _BOOK_INDEX.items() if corpus_id in corpora),
        key=lambda name: list(_BOOK_INDEX).index(name),
    )


def canonical_book(corpus_id: str, name: str) -> Tuple[str, str]:
    """Resolve a source's book name to (book_id, canonical display name).

    Raises CorpusFormatError when the name is not a book of that corpus, which is
    how a malformed heading or an unexpected body shows up instead of silently
    producing records under the wrong book.
    """
    cleaned = re.sub(r"\s+", " ", name).strip()
    cleaned = BOOK_ALIASES.get(cleaned, cleaned)

    corpora = _BOOK_INDEX.get(cleaned)
    if not corpora or corpus_id not in corpora:
        raise CorpusFormatError(
            f"{cleaned!r} is not a known book of corpus {corpus_id!r}"
        )

    return slugify(cleaned), cleaned


# -------------------------------------------------
# Canonical record
# -------------------------------------------------

def make_verse(corpus_id: str,
               book: str,
               chapter: int,
               verse: int,
               text: str,
               source_ref: str,
               source_line: int) -> Dict:
    book_id, canonical_name = canonical_book(corpus_id, book)

    if not isinstance(chapter, int) or chapter < 1:
        raise CorpusFormatError(f"bad chapter {chapter!r} at {source_ref}")
    if not isinstance(verse, int) or verse < 1:
        raise CorpusFormatError(f"bad verse {verse!r} at {source_ref}")

    return {
        "corpus": corpus_id,
        "book_id": book_id,
        "book": canonical_name,
        "chapter": chapter,
        "verse": verse,
        "text": text,
        "source_ref": source_ref,
        "source_line": source_line,
    }


def format_reference(book: str, chapter: int, verse: int, end_verse: Optional[int] = None) -> str:
    """Human-readable reference; ranges collapse to 'Isaiah 53:1-12'."""
    if end_verse is not None and end_verse != verse:
        return f"{book} {chapter}:{verse}-{end_verse}"
    return f"{book} {chapter}:{verse}"


# -------------------------------------------------
# Book of Mormon (Project Gutenberg layout)
# -------------------------------------------------

_ORDINALS = {"FIRST": "1", "SECOND": "2", "THIRD": "3", "FOURTH": "4"}

_TITLE_RE = re.compile(
    r'^\s*(?:THE\s+)?(?:(?P<ordinal>FIRST|SECOND|THIRD|FOURTH)\s+)?'
    r'(?:BOOK\s+OF\s+)?(?P<title>[A-Z]+(?:\s+[A-Z]+)*)'
    r'\s*(?:\((?P<abbr>[^)]+)\))?\s*$'
)

_CHAPTER_HEADER_RE = re.compile(
    r'^(?:(?P<number>[1-4])\s+)?(?P<book>[A-Z][A-Za-z]+)\s+Chapter\s+(?P<chapter>\d+)\s*$'
)

_VERSE_RE = re.compile(r'^(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.*)$')

_INLINE_VERSE_RE = re.compile(
    r'^(?P<number>[1-4])\s+(?P<book>[A-Z][A-Za-z]+)\s+'
    r'(?P<chapter>\d+):(?P<verse>\d+)\s+(?P<text>.*)$'
)

_TITLE_NAMES = {
    "NEPHI": "Nephi", "JACOB": "Jacob", "ENOS": "Enos", "JAROM": "Jarom",
    "OMNI": "Omni", "WORDS OF MORMON": "Words of Mormon", "MOSIAH": "Mosiah",
    "ALMA": "Alma", "HELAMAN": "Helaman", "MORMON": "Mormon",
    "ETHER": "Ether", "MORONI": "Moroni",
}


def _bom_title_book(title: str, ordinal: Optional[str], abbr: Optional[str]) -> Optional[str]:
    if abbr:
        return abbr.strip()
    base = _TITLE_NAMES.get(title.strip().upper())
    if base is None:
        return None
    return f"{_ORDINALS[ordinal]} {base}" if ordinal else base


def _bom_is_title_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return "BOOK OF" in stripped.upper() or (
        stripped.upper() == stripped and any(ch.isalpha() for ch in stripped)
    )


def parse_bom_text(text: str) -> List[Dict]:
    """Parse the Gutenberg Book of Mormon into canonical verse records."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start = None
    for index, line in enumerate(lines):
        if _CHAPTER_HEADER_RE.match(line.strip()):
            start = index
            break
        title_match = _TITLE_RE.match(line) if _bom_is_title_line(line) else None
        if title_match and _bom_title_book(
            title_match.group("title"), title_match.group("ordinal"), title_match.group("abbr")
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
            current["text"] = re.sub(r"\s+", " ", current["text"]).strip()
            verses.append(make_verse(
                "bom", current["book"], current["chapter"], current["verse"],
                current["text"], current["source_ref"], current["source_line"],
            ))
            current = None

    for offset, raw_line in enumerate(lines[start:]):
        line_no = start + offset + 1
        line = raw_line.strip()

        if _bom_is_title_line(raw_line):
            title_match = _TITLE_RE.match(raw_line)
            if title_match:
                candidate = _bom_title_book(
                    title_match.group("title"),
                    title_match.group("ordinal"),
                    title_match.group("abbr"),
                )
                if candidate:
                    close_current()
                    book, chapter = candidate, 1
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
            verse = int(inline_match.group("verse"))
            current = {
                "book": book, "chapter": chapter, "verse": verse,
                "text": inline_match.group("text"), "source_line": line_no,
                "source_ref": format_reference(book, chapter, verse),
            }
            continue

        verse_match = _VERSE_RE.match(line)
        if verse_match and book is not None and chapter is not None:
            close_current()
            verse = int(verse_match.group("verse"))
            current = {
                "book": book, "chapter": chapter, "verse": verse,
                "text": verse_match.group("text"), "source_line": line_no,
                "source_ref": format_reference(book, chapter, verse),
            }
            continue

        if current is not None and line:
            current["text"] += " " + line

    close_current()
    return verses


# -------------------------------------------------
# King James Bible (one tab-separated verse per line)
# -------------------------------------------------

_KJV_LINE_RE = re.compile(
    r'^(?P<book>[1-3]?\s?[A-Za-z][A-Za-z ]*?)\s+'
    r'(?P<chapter>\d+):(?P<verse>\d+)\t(?P<text>.*)$'
)


def parse_kjv_text(text: str) -> List[Dict]:
    """Parse 'Book C:V<TAB>text' lines into canonical verse records.

    Header lines and blank lines are skipped; a line that is neither a header nor
    a verse raises CorpusFormatError rather than being dropped, so a truncated or
    re-encoded file cannot look like a shorter canon.
    """
    verses: List[Dict] = []

    for line_no, raw_line in enumerate(text.split("\n"), 1):
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue

        match = _KJV_LINE_RE.match(line)
        if not match:
            if line_no <= 3 and not re.search(r"\d+:\d+", line):
                continue  # corpus title / provenance lines
            raise CorpusFormatError(
                f"line {line_no} is not a verse or a header: {line[:80]!r}"
            )

        book = match.group("book").strip()
        chapter = int(match.group("chapter"))
        verse = int(match.group("verse"))
        verses.append(make_verse(
            "kjv", book, chapter, verse, match.group("text"),
            format_reference(book, chapter, verse), line_no,
        ))

    if not verses:
        raise CorpusFormatError("no verses found in KJV source")

    return verses


# -------------------------------------------------
# Corpus registry
# -------------------------------------------------

CORPUS_PARSERS = {
    "bom": parse_bom_text,
    "kjv": parse_kjv_text,
}


def available_corpora() -> List[str]:
    return sorted(CORPUS_PARSERS)


def load_corpus_text(corpus_id: str, text: str) -> List[Dict]:
    parser = CORPUS_PARSERS.get(corpus_id)
    if parser is None:
        raise ValueError(
            f"unknown corpus {corpus_id!r}; known: {', '.join(available_corpora())}"
        )
    return parser(text)


def load_corpus(corpus_id: str, path) -> List[Dict]:
    raw = Path(path).read_text(encoding="utf-8-sig")
    return load_corpus_text(corpus_id, raw)


def corpus_digest(path) -> str:
    """sha256 of the source file, recorded with run metadata for reproducibility."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
