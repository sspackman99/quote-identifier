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


# Book ids readers actually use, where the slug would be ugly ("d-c" for D&C).
BOOK_ID_OVERRIDES = {"D&C": "dc"}


def book_id_for(name: str) -> str:
    return BOOK_ID_OVERRIDES.get(name.strip(), slugify(name))


def _register(corpus_id: str, names: List[str]) -> None:
    for name in names:
        _BOOK_INDEX.setdefault(name, {})[corpus_id] = book_id_for(name)


_register("bom", BOM_BOOKS)
_register("kjv", KJV_BOOKS)

# Doctrine and Covenants is one book of sections: the section number is carried in
# the `chapter` field, so refs read "D&C 76:22" rather than a paragraph number.
DC_BOOKS = ["D&C"]
# Pearl of Great Price in its canonical order. It is several books, and references
# are written with the book name: "Moses 1:1", "Abraham 3:25".
PGP_BOOKS = ["Moses", "Abraham", "Joseph Smith-Matthew", "Joseph Smith-History"]

_register("dc", DC_BOOKS)
_register("pgp", PGP_BOOKS)

BOOK_ALIASES.update({
    "Doctrine and Covenants": "D&C",
    "D and C": "D&C",
    "Joseph Smith - Matthew": "Joseph Smith-Matthew",
    "Joseph Smith—Matthew": "Joseph Smith-Matthew",
    "Joseph Smith - History": "Joseph Smith-History",
    "Joseph Smith—History": "Joseph Smith-History",
})


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

    return book_id_for(cleaned), cleaned


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
    # The licence block after the canon is not scripture. Left in, it is appended
    # to the last verse (Moroni 10:34), turning one verse into ~350 lines and
    # feeding "project gutenberg" windows into retrieval and into the classifier.
    end_marker = re.search(
        r"^\*\*\*\s*END OF THE PROJECT GUTENBERG.*$", text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if end_marker:
        text = text[:end_marker.start()]

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
# Doctrine and Covenants (sacred-texts edition)
# -------------------------------------------------

_DC_VERSE_RE = re.compile(r"^\[(?P<section>\d{1,3}):(?P<verse>\d{1,3})\]\s*(?P<text>.*)$")
# A verse whose bracket marker was lost looks like "76:22 And the Lord said…".
_DC_BARE_VERSE_RE = re.compile(r"^\d{1,3}:\d{1,3}\s")


def parse_dc_text(text: str) -> List[Dict]:
    """Parse the sacred-texts Doctrine and Covenants: '[section:verse] text'.

    Verse markers are authoritative here. The edition's section headings are page
    commentary — "1. Revelation through Joseph Smith, Hiram, Ohio, November 1831."
    — and one of them, section 78, has lost its number entirely, so headings
    cannot establish section identity; the bracket markers can.

    A line that starts with "[" and does not parse raises, because that is what a
    changed verse format looks like. Heading and footnote lines between verses are
    recognised by shape (they end with a full stop); anything else raises rather
    than being guessed at.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    start = next(
        (index for index, line in enumerate(lines) if _DC_VERSE_RE.match(line.strip())),
        None,
    )
    if start is None:
        raise CorpusFormatError("no '[section:verse]' marker found in the text")

    records = []
    for number, raw in enumerate(lines[start:], start=start + 1):
        line = raw.strip()
        if not line:
            continue

        match = _DC_VERSE_RE.match(line)
        if match:
            records.append(make_verse(
                "dc", book="D&C",
                chapter=int(match.group("section")),
                verse=int(match.group("verse")),
                text=match.group("text").strip(),
                source_ref=f"[{match.group('section')}:{match.group('verse')}]",
                source_line=number,
            ))
            continue

        if line.startswith("["):
            raise CorpusFormatError(
                f"unparseable verse marker at line {number}: {line[:80]!r}"
            )
        if _DC_BARE_VERSE_RE.match(line):
            raise CorpusFormatError(
                f"verse-like line without a bracket marker at line {number}: "
                f"{line[:80]!r}"
            )
        # otherwise: an edition heading or footnote for the section around it.
        # Headings wrap across lines ("…July 1830. See" / "History of the Church
        # 1:104-8."), so their shape cannot be pinned down more tightly than this.

    return records


# -------------------------------------------------
# Pearl of Great Price (sacred-texts edition)
# -------------------------------------------------

_PGP_HEADER_RE = re.compile(
    r"^Pearl of Great Price:\s*(?P<book>[^:]+):\s*Chapter\s+(?P<chapter>\d+)\s*$"
)
_PGP_VERSE_RE = re.compile(r"^(?P<verse>\d{1,3})\s+(?P<text>\S.*)$")
# Some verses put the number on its own line, with the text on the next one.
_PGP_BARE_VERSE_RE = re.compile(r"^(?P<verse>\d{1,3})$")
_PGP_NAV_LINES = {"Sacred Texts", "|", "Mormonism", "Index"}


def _repair_mojibake(text: str) -> str:
    """Undo the double encoding in this scrape ('bowsÃ·and', 'Godâ€™s').

    The bytes are UTF-8 decoded as cp1252/latin-1 once too often, so re-encoding
    to the single-byte codec and decoding as UTF-8 recovers the intended
    characters. cp1252 is tried first because it is what produced the 'â€™'
    sequences; latin-1 covers the rest.
    """
    if not any(marker in text for marker in ("Ã", "â", "Â")):
        return text
    for codec in ("cp1252", "latin-1"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def parse_pgp_text(text: str) -> List[Dict]:
    """Parse the sacred-texts Pearl of Great Price.

    The scrape interleaves navigation ("Sacred Texts | Mormonism", "Next: … »")
    with the text; verses are hard-wrapped, so a line without a number continues
    the verse above it; and the whole Joseph Smith-History block appears twice, so
    the first copy is kept and later copies of an already-seen chapter are
    skipped.

    Gaps in the verse numbering are preserved as they are — this edition really is
    missing Joseph Smith-History 37 and 69 — while a repeated verse number inside
    one chapter raises. Text is repaired from the file's double encoding, which
    otherwise puts "Ã·" in the middle of verses.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    records = []
    pending = None
    book = None
    chapter = None
    seen_chapters = set()
    seen_verses = set()
    ignoring_duplicate = False

    def flush():
        nonlocal pending
        if pending is None:
            return
        body = _repair_mojibake(pending["text"]).strip()
        if not body:
            raise CorpusFormatError(
                f"verse {pending['book']} {pending['chapter']}:{pending['verse']} "
                f"has no text (line {pending['line']})"
            )
        records.append(make_verse(
            "pgp", book=pending["book"], chapter=pending["chapter"],
            verse=pending["verse"], text=body,
            source_ref=f"{pending['book']} {pending['chapter']}:{pending['verse']}",
            source_line=pending["line"],
        ))
        pending = None

    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue

        header = _PGP_HEADER_RE.match(line)
        if header:
            flush()
            name = header.group("book").strip()
            chapter_number = int(header.group("chapter"))
            key = (name, chapter_number)

            if key in seen_chapters:
                ignoring_duplicate = True
                book, chapter = None, None
                continue

            seen_chapters.add(key)
            ignoring_duplicate = False
            book, chapter = name, chapter_number
            seen_verses = set()
            continue

        if line in _PGP_NAV_LINES or line.startswith(("« Previous", "Next:", "Previous:")):
            continue
        if ignoring_duplicate:
            continue

        if book is None:
            raise CorpusFormatError(
                f"verse outside a chapter heading at line {number}: {line[:60]!r}"
            )

        numbered = _PGP_VERSE_RE.match(line)
        bare = _PGP_BARE_VERSE_RE.match(line)

        if numbered or bare:
            flush()
            verse_number = int((numbered or bare).group("verse"))
            if verse_number in seen_verses:
                raise CorpusFormatError(
                    f"verse {book} {chapter}:{verse_number} appears twice "
                    f"(line {number})"
                )
            seen_verses.add(verse_number)
            pending = {
                "book": book, "chapter": chapter, "verse": verse_number,
                "text": numbered.group("text") if numbered else "",
                "line": number,
            }
            continue

        if pending is None:
            raise CorpusFormatError(
                f"unrecognised line with no verse to continue at line {number}: "
                f"{line[:60]!r}"
            )
        pending["text"] = f"{pending['text']} {line}".strip()

    flush()
    return records


# -------------------------------------------------
# Corpus registry
# -------------------------------------------------

CORPUS_PARSERS = {
    "bom": parse_bom_text,
    "kjv": parse_kjv_text,
    "dc": parse_dc_text,
    "pgp": parse_pgp_text,
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
