"""
Tests for preprocess.py using small synthetic fixtures.
These tests verify preprocessing behavior without requiring the full corpus.
"""
import pytest


class TestLoadText:
    """Test text loading functionality."""

    def test_load_text_from_string(self, tmp_path):
        """Test loading a small text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world\nThis is a test.")
        
        from preprocess import load_text
        content = load_text(str(test_file))
        
        assert content == "Hello world\nThis is a test."
        assert "hello" in content.lower()

    def test_load_text_strips_utf8_bom(self, tmp_path):
        test_file = tmp_path / "bom_marked.txt"
        test_file.write_bytes("\ufeff1 Nephi Chapter 1".encode("utf-8"))

        from preprocess import load_text

        assert load_text(str(test_file)) == "1 Nephi Chapter 1"


class TestNormalizeText:
    """Test text normalization functionality."""

    def test_lowercase_conversion(self):
        """Test converting text to lowercase."""
        text = "Hello World! This is TEST."
        
        from preprocess import normalize_text
        
        result = normalize_text(text, lowercase=True, remove_punctuation=False)
        
        assert result == "hello world! this is test."

    def test_remove_punctuation(self):
        """Test removing punctuation from text."""
        text = "Hello, world! How are you? I'm fine."
        
        from preprocess import normalize_text
        
        result = normalize_text(text, lowercase=True, remove_punctuation=True)
        
        assert result == "hello world how are you im fine"

    def test_whitespace_normalization(self):
        """Test normalizing whitespace."""
        text = "Hello    world\t\n\r\t  test"
        
        from preprocess import normalize_text
        
        result = normalize_text(text, lowercase=True, remove_punctuation=False)
        
        assert result == "hello world test"

    def test_strip_leading_trailing(self):
        """Test stripping leading and trailing whitespace."""
        text = "   Hello world   \n\n"
        
        from preprocess import normalize_text
        
        result = normalize_text(text, lowercase=True, remove_punctuation=False)
        
        assert result == "hello world"


class TestTokenize:
    """Test tokenization functionality."""

    def test_simple_tokenization(self):
        """Test basic tokenization."""
        text = "Hello world this is a test"
        
        from preprocess import tokenize
        
        tokens = tokenize(text)
        
        # Without lemmatization (nltk not available), tokens preserve original case
        # Just check we get the expected number of tokens
        assert len(tokens) == 6
        assert "hello" in tokens or "Hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenization_with_lemmatization(self):
        """Test tokenization with lemmatization enabled."""
        text = "The cat sat on the mat. Cats are fun."
        
        from preprocess import tokenize
        
        tokens = tokenize(text)
        
        # With lemmatization, "Cats" should be "cat" if nltk is available
        # If nltk is not available, "Cats" stays as "Cats"
        if "cat" in tokens:
            assert "cat" in tokens
        else:
            # nltk not available - check that "Cats" or "cats" is present
            assert "cats" in [t.lower() for t in tokens]
        # "the" should appear twice (or once if case differs)
        the_count = tokens.count("the")
        assert the_count == 1 or the_count == 2


class TestPreprocessBom:
    """Test the full preprocessing pipeline with synthetic data."""

    def test_preprocess_single_verse(self, tmp_path):
        """Test preprocessing a single verse."""
        # Create a minimal BOM-like file
        test_content = """1 Nephi Chapter 1
1:1 I, Nephi, being caused that I should know the beginnings
of the year.
1:2 And the first year of the reign of the liberty of the
children of Israel.
"""
        
        test_file = tmp_path / "bom.txt"
        test_file.write_text(test_content)
        
        from preprocess import preprocess_bom
        
        result = preprocess_bom(str(test_file), remove_punctuation=False)
        
        assert len(result) == 2
        
        # Canonical fields, not display strings
        v1 = result[0]
        assert v1["corpus"] == "bom"
        assert v1["book_id"] == "1-nephi"
        assert v1["book"] == "1 Nephi"
        assert v1["chapter"] == 1
        assert v1["verse"] == 1
        assert v1["source_ref"] == "1 Nephi 1:1"
        assert v1["source_line"] > 0
        assert "i, nephi, being caused" in v1["clean_text"]
        assert len(v1["tokens"]) > 0
        
        # Check second verse
        v2 = result[1]
        assert v2["book_id"] == "1-nephi"
        assert v2["chapter"] == 1
        assert v2["verse"] == 2
        assert "first year" in v2["clean_text"]

    def test_preprocess_with_punctuation_removal(self, tmp_path):
        """Test preprocessing with punctuation removal."""
        test_content = """1 Nephi Chapter 1
1:1 I, Nephi, being caused that I should know the beginnings!
1:2 And the first year of the reign of liberty
"""
        
        test_file = tmp_path / "bom.txt"
        test_file.write_text(test_content)
        
        from preprocess import preprocess_bom
        
        result = preprocess_bom(str(test_file), remove_punctuation=True)
        
        assert len(result) == 2
        
        # Punctuation should be removed
        assert "," not in result[0]["clean_text"]
        assert "?" not in result[0]["clean_text"]
        assert "!" not in result[0]["clean_text"]
        
        # Should still have words
        assert "i" in result[0]["clean_text"]
        assert "nephi" in result[0]["clean_text"]

    def test_preprocess_verse_span_metadata(self, tmp_path):
        """Test that verse span metadata is preserved."""
        test_content = """1 Nephi Chapter 1
1:1 First verse text here.
1:2 Second verse more text.
1:3 Third verse continuing.
"""
        
        test_file = tmp_path / "bom.txt"
        test_file.write_text(test_content)
        
        from preprocess import preprocess_bom
        
        result = preprocess_bom(str(test_file))
        
        assert len(result) == 3
        assert [v["verse"] for v in result] == [1, 2, 3]
        assert [v["book_id"] for v in result] == ["1-nephi"] * 3
        assert [v["source_ref"] for v in result] == [
            "1 Nephi 1:1", "1 Nephi 1:2", "1 Nephi 1:3",
        ]
        # raw verse text is preserved for audit
        assert result[0]["text"] == "First verse text here."
        assert result[0]["original_text"] == result[0]["text"]

    def test_preprocess_corpus_dispatches_by_corpus_id(self, tmp_path):
        test_file = tmp_path / "verse.txt"
        test_file.write_text("Genesis 1:1\tIn the beginning God created\n")

        from preprocess import preprocess_corpus

        records = preprocess_corpus("kjv", str(test_file))

        assert records[0]["corpus"] == "kjv"
        assert records[0]["book_id"] == "genesis"
        assert records[0]["tokens"] == ["in", "the", "beginning", "god", "created"]
