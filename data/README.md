# Corpus data (not tracked)

The scripture text this pipeline runs on is **not** stored in this repository.
`data/` is where you put your own copy.

## Why it is not in git

The local copy is the Project Gutenberg edition of the Book of Mormon, and that
file carries Gutenberg's own header and licence terms. Keeping it out of the
repository keeps redistribution a deliberate decision instead of a side effect
of a commit — and the text is 1.5 MB of replaceable data, not source.

## How to supply it

Any one of these, in priority order:

    python main.py --corpus /path/to/bom.txt
    QI_CORPUS=/path/to/bom.txt python main.py
    cp /path/to/bom.txt data/bom.txt && python main.py

With no argument, `main.py` looks for `data/bom.txt`, then `./bom.txt`, and
exits with a message naming every location it checked if neither exists.

## Fetching a text

The Book of Mormon is available from Project Gutenberg
(https://www.gutenberg.org/ebooks/17). Preserve the file's front matter if you
plan to redistribute it; the parser skips everything before the first chapter
heading, so the licence block does no harm.

## Expected format

`preprocess.py` understands the Project Gutenberg layout:

    THE FIRST BOOK OF NEPHI HIS REIGN AND MINISTRY (1 Nephi)

    1 Nephi Chapter 1

    1:1 I, Nephi, having been born of goodly parents, ...

- book titles: `THE BOOK OF MOSIAH`, `THE SECOND BOOK OF NEPHI`, `FOURTH NEPHI`
- chapter headings: `Mosiah Chapter 12`, `1 Nephi Chapter 1`
- verses: `N:N text`, continuing until the next verse marker; single-chapter
  books (Enos, Jarom, Omni) have no chapter heading
- inline verse markers: `4 Nephi 1:1 text`

A text in a different layout (Bible chapters without verse numbers, Doctrine and
Covenants sections, Pearl of Great Price books) will parse as one book with no
verses, or raise nothing at all — check the verse count with
`python -m pytest test_parse_scripture.py` (the real-corpus guards skip when no
corpus file is present) before trusting a run.

## What the local edition contains

The Gutenberg text used here parses to 14 books, 238 chapters, 6,598 verses.
It has no separate Words of Mormon section, and 4 Nephi appears only through its
inline `4 Nephi 1:N` verse markers.
