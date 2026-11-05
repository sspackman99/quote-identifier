"""
Tests for chunking.py using small synthetic fixtures.
These tests verify chunking behavior without requiring the full corpus.
"""
import pytest


class TestBuildChapterTokenStream:
    """Test building continuous token streams per chapter."""

    def test_single_chapter(self):
        """Test with a single chapter."""
        from chunking import build_chapter_token_stream
        
        processed_verses = [
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:1",
                "original_text": "I, Nephi, being caused.",
                "clean_text": "i, nephi, being caused.",
                "tokens": ["i", "nephi", "being", "caused"]
            },
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:2",
                "original_text": "The first year.",
                "clean_text": "the first year.",
                "tokens": ["the", "first", "year"]
            }
        ]
        
        streams = build_chapter_token_stream(processed_verses)
        
        assert len(streams) == 1
        assert streams[0]["book"] == "1 Nephi"
        assert streams[0]["chapter"] == 1
        assert len(streams[0]["tokens"]) == 7
        assert "i" in streams[0]["tokens"][0]
        
        # Verify token_to_verse mapping
        assert len(streams[0]["token_to_verse"]) == 7
        assert streams[0]["token_to_verse"][0] == "1:1"
        assert streams[0]["token_to_verse"][4] == "1:2"

    def test_multiple_chapters(self):
        """Test with multiple chapters."""
        from chunking import build_chapter_token_stream
        
        processed_verses = [
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:1",
                "original_text": "First verse.",
                "clean_text": "first verse.",
                "tokens": ["first", "verse"]
            },
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 2,
                "verse": "2:1",
                "original_text": "Second verse.",
                "clean_text": "second verse.",
                "tokens": ["second", "verse"]
            },
            {
                "corpus": "bom",
                "book": "2 Nephi",
                "chapter": 1,
                "verse": "1:1",
                "original_text": "Third verse.",
                "clean_text": "third verse.",
                "tokens": ["third", "verse"]
            }
        ]
        
        streams = build_chapter_token_stream(processed_verses)
        
        assert len(streams) == 3
        
        books_chapters = [(s["book"], s["chapter"]) for s in streams]
        assert streams[0]["book"] == "1 Nephi"
        assert streams[0]["chapter"] == 1
        assert streams[1]["book"] == "1 Nephi"
        assert streams[1]["chapter"] == 2
        assert streams[2]["book"] == "2 Nephi"

    def test_token_stream_continuity(self):
        """Test that tokens within a chapter form a continuous stream."""
        from chunking import build_chapter_token_stream
        
        processed_verses = [
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:1",
                "original_text": "A B C.",
                "clean_text": "a b c.",
                "tokens": ["a", "b", "c"]
            },
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:2",
                "original_text": "D E F.",
                "clean_text": "d e f.",
                "tokens": ["d", "e", "f"]
            },
            {
                "corpus": "bom",
                "book": "1 Nephi",
                "chapter": 1,
                "verse": "1:3",
                "original_text": "G H I.",
                "clean_text": "g h i.",
                "tokens": ["g", "h", "i"]
            }
        ]
        
        streams = build_chapter_token_stream(processed_verses)
        
        assert len(streams) == 1
        chapter_stream = streams[0]
        
        # All 9 tokens should be in one continuous stream
        assert len(chapter_stream["tokens"]) == 9
        assert chapter_stream["tokens"] == ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        
        # Token to verse mapping should be correct
        assert chapter_stream["token_to_verse"] == ["1:1", "1:1", "1:1", "1:2", "1:2", "1:2", "1:3", "1:3", "1:3"]


class TestCreateOverlappingWindows:
    """Test creating overlapping windows from chapter streams."""

    def test_windows_with_overlap(self):
        """Test creating windows with specified overlap."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
            "token_to_verse": ["1:1", "1:1", "1:2", "1:2", "1:3", "1:3", "1:3", "1:4", "1:4", "1:4"]
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=2)
        
        # step = 3 - 2 = 1
        # With 10 tokens and window_size=3, step=1: windows at 0, 1, 2, 3, 4, 5, 6, 7
        assert len(windows) == 8
        
        # First window: tokens 0-2 (indices), tokens 0-1 are verse 1:1, token 2 is verse 1:2
        assert windows[0]["start_token_index"] == 0
        assert windows[0]["end_token_index"] == 3
        assert windows[0]["tokens"] == ["a", "b", "c"]
        assert windows[0]["start_verse"] == "1:1"
        assert windows[0]["end_verse"] == "1:2"
        
        # Second window: tokens 1-3
        assert windows[1]["start_token_index"] == 1
        assert windows[1]["end_token_index"] == 4
        assert windows[1]["tokens"] == ["b", "c", "d"]
        assert windows[1]["start_verse"] == "1:1"
        assert windows[1]["end_verse"] == "1:2"

    def test_windows_with_no_overlap(self):
        """Test creating windows with no overlap (step = window_size)."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": ["a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "token_to_verse": ["1:1"] * 9
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=0)
        
        # step = 3 - 0 = 3
        assert len(windows) == 3
        
        assert windows[0]["tokens"] == ["a", "b", "c"]
        assert windows[1]["tokens"] == ["d", "e", "f"]
        assert windows[2]["tokens"] == ["g", "h", "i"]

    def test_windows_spanning_multiple_verses(self):
        """Test that windows correctly span verse boundaries."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": ["a", "b", "c", "d", "e"],
            "token_to_verse": ["1:1", "1:1", "1:2", "1:2", "1:2"]
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=2)
        
        # step = 3 - 2 = 1
        # range(0, 5 - 3 + 1, 1) = range(0, 3, 1) = [0, 1, 2] = 3 windows
        assert len(windows) == 3
        
        # Window 0: tokens 0-2 (indices), spans verses 1:1 and 1:2
        assert windows[0]["start_verse"] == "1:1"
        assert windows[0]["end_verse"] == "1:2"
        
        # Window 1: tokens 1-3
        assert windows[1]["start_verse"] == "1:1"
        assert windows[1]["end_verse"] == "1:2"
        
        # Window 2: tokens 2-4
        assert windows[2]["start_verse"] == "1:2"
        assert windows[2]["end_verse"] == "1:2"

    def test_windows_metadata_preservation(self):
        """Test that window metadata is preserved correctly."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": ["a", "b", "c"],
            "token_to_verse": ["1:1", "1:1", "1:1"]
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=2)
        
        assert len(windows) == 1
        w = windows[0]
        
        assert w["corpus"] == "bom"
        assert w["book"] == "1 Nephi"
        assert w["chapter"] == 1
        assert w["start_token_index"] == 0
        assert w["end_token_index"] == 3
        assert w["start_verse"] == "1:1"
        assert w["end_verse"] == "1:1"
        assert w["tokens"] == ["a", "b", "c"]
        assert w["text"] == "a b c"

    def test_empty_chapter(self):
        """Test handling of empty chapter (no tokens)."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": [],
            "token_to_verse": []
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=2)
        
        assert len(windows) == 0

    def test_short_chapter_larger_than_window(self):
        """Test chapter with fewer tokens than window size."""
        from chunking import create_overlapping_windows
        
        chapter_data = {
            "corpus": "bom",
            "book": "1 Nephi",
            "chapter": 1,
            "tokens": ["a", "b"],  # Only 2 tokens
            "token_to_verse": ["1:1"] * 2
        }
        
        windows = create_overlapping_windows([chapter_data], window_size=3, overlap=2)
        
        # A chapter shorter than the window is one window holding the whole chapter
        assert len(windows) == 1
        assert windows[0]["tokens"] == ["a", "b"]
        assert windows[0]["start_token_index"] == 0
        assert windows[0]["end_token_index"] == 2
