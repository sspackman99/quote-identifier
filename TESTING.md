# Test Foundation for Quote-Identifier

This directory contains pytest-based unit tests for the quote-identification pipeline, focusing on deterministic code that does not require downloading models or running the full corpus.

## Tests

### test_preprocess.py

Tests for `preprocess.py` covering:

1. **TestLoadText** - Text loading from files
2. **TestSplitBookChapters** - Splitting text into book/chapter sections
3. **TestParseHeader** - Parsing book/chapter headers
4. **TestSplitVerses** - Splitting chapter bodies into verses
5. **TestNormalizeText** - Text normalization (lowercase, punctuation removal, whitespace normalization)
6. **TestTokenize** - Tokenization with and without lemmatization
7. **TestPreprocessBom** - Full preprocessing pipeline with synthetic fixtures

### test_chunking.py

Tests for `chunking.py` covering:

1. **TestBuildChapterTokenStream** - Building continuous token streams per chapter
   - Single chapter
   - Multiple chapters
   - Token stream continuity verification

2. **TestCreateOverlappingWindows** - Creating overlapping windows from chapter streams
   - Windows with specified overlap
   - Windows with no overlap
   - Windows spanning multiple verses
   - Metadata preservation
   - Edge cases (empty chapter, short chapter)

## Running Tests

```bash
# Run all tests
cd /path/to/quote-identifier/.worktrees/t_01172f03
python3 -m pytest test_preprocess.py test_chunking.py -v

# Run with coverage
python3 -m pytest test_preprocess.py test_chunking.py --cov=. --cov-report=html

# Run specific test file
python3 -m pytest test_preprocess.py -v
```

## Requirements

- Python 3.11+
- pytest
- nltk (optional, for lemmatization tests - tests gracefully handle when nltk is unavailable)

## Notes

- All tests use small synthetic fixtures instead of the complete corpus
- Tests verify preprocessing behavior including verse-span metadata
- Tests cover overlapping sliding-window behavior
- Production behavior is unchanged (only test files added)
- Tests are deterministic and do not depend on external model downloads

## Test Results

All 23 tests pass successfully:
- 14 tests in test_preprocess.py
- 9 tests in test_chunking.py
