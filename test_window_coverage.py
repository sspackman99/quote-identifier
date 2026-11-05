# test_window_coverage.py
"""Coverage properties for create_overlapping_windows.

These are checked geometrically (which token indices each window covers),
independently of the similarity pipeline, so a windowing change is validated on
its own terms.
"""

import pytest

from chunking import create_overlapping_windows

# (window_size, overlap) pairs used by the sweeps, including the pipeline's own
# settings (30/15) and coarse/fine variants.
PARAMS = [
    (30, 15),
    (10, 9),
    (10, 5),
    (10, 0),
    (4, 2),
    (3, 2),
    (3, 0),
    (2, 1),
]


def chapter_stream(book, chapter, total_tokens):
    return {
        "corpus": "bom",
        "book": book,
        "chapter": chapter,
        "tokens": [f"{book}-{chapter}-t{i}" for i in range(total_tokens)],
        "token_to_verse": [f"{chapter}:{i // 5 + 1}" for i in range(total_tokens)],
    }


def coverage(windows, stream):
    """Token indices of a chapter covered by the windows built from it."""
    covered = set()
    for window in windows:
        if (window["book"], window["chapter"]) != (stream["book"], stream["chapter"]):
            continue
        covered.update(range(window["start_token_index"], window["end_token_index"]))
    return covered


@pytest.mark.parametrize("window_size,overlap", PARAMS)
@pytest.mark.parametrize("total", [0, 1, 2, 3, 4, 5, 9, 10, 11, 29, 30, 31, 47])
def test_every_token_is_covered(window_size, overlap, total):
    """Property 5: every token in a chapter appears in at least one window."""
    stream = chapter_stream("1 Nephi", 1, total)
    windows = create_overlapping_windows([stream], window_size=window_size, overlap=overlap)

    assert coverage(windows, stream) == set(range(total))


@pytest.mark.parametrize("window_size,overlap", PARAMS)
@pytest.mark.parametrize("total", [0, 1, 2, 4, 5, 10, 31])
def test_no_window_crosses_a_chapter_boundary(window_size, overlap, total):
    """Property 3: each window belongs to exactly one chapter, over its own tokens."""
    first = chapter_stream("1 Nephi", 1, total)
    second = chapter_stream("2 Nephi", 1, total)
    windows = create_overlapping_windows([first, second], window_size=window_size, overlap=overlap)

    for window in windows:
        source = first if window["book"] == "1 Nephi" else second
        assert window["corpus"] == source["corpus"]
        assert window["chapter"] == source["chapter"]
        assert window["tokens"] == source["tokens"][
            window["start_token_index"]:window["end_token_index"]
        ]
        assert window["text"] == " ".join(window["tokens"])


@pytest.mark.parametrize("window_size,overlap", PARAMS)
@pytest.mark.parametrize("total", [1, 2, 3, 5, 10, 30, 31, 47])
def test_no_duplicate_windows(window_size, overlap, total):
    """Property 4: a chapter never yields the same span twice."""
    stream = chapter_stream("1 Nephi", 1, total)
    windows = create_overlapping_windows([stream], window_size=window_size, overlap=overlap)

    spans = [(w["start_token_index"], w["end_token_index"]) for w in windows]
    assert len(spans) == len(set(spans))
    token_tuples = [tuple(w["tokens"]) for w in windows]
    assert len(token_tuples) == len(set(token_tuples))


@pytest.mark.parametrize("window_size,overlap", PARAMS)
@pytest.mark.parametrize("total", [1, 2, 5, 10, 30, 31])
def test_window_never_exceeds_window_size(window_size, overlap, total):
    stream = chapter_stream("1 Nephi", 1, total)
    windows = create_overlapping_windows([stream], window_size=window_size, overlap=overlap)

    for window in windows:
        assert len(window["tokens"]) <= window_size
        assert window["end_token_index"] - window["start_token_index"] == len(window["tokens"])


def test_short_chapter_becomes_one_whole_chapter_window():
    """Property 1: a chapter shorter than the window is one window, not zero."""
    stream = chapter_stream("1 Nephi", 1, 2)
    windows = create_overlapping_windows([stream], window_size=3, overlap=2)

    assert len(windows) == 1
    assert windows[0]["start_token_index"] == 0
    assert windows[0]["end_token_index"] == 2
    assert windows[0]["tokens"] == stream["tokens"]
    assert windows[0]["text"] == " ".join(stream["tokens"])


def test_chapter_equal_to_window_is_one_window():
    stream = chapter_stream("1 Nephi", 1, 4)
    windows = create_overlapping_windows([stream], window_size=4, overlap=3)

    assert len(windows) == 1
    assert windows[0]["end_token_index"] == 4


def test_tail_window_is_anchored_to_chapter_end():
    """Property 2: a chapter ending off the step grid gets an end-anchored window."""
    stream = chapter_stream("1 Nephi", 1, 10)
    windows = create_overlapping_windows([stream], window_size=3, overlap=1)  # step 2

    starts = [w["start_token_index"] for w in windows]
    assert starts == [0, 2, 4, 6, 7], starts
    assert windows[-1]["end_token_index"] == 10
    assert windows[-1]["tokens"] == stream["tokens"][7:10]


def test_tail_window_is_not_added_when_grid_already_ends_at_chapter_end():
    stream = chapter_stream("1 Nephi", 1, 9)
    windows = create_overlapping_windows([stream], window_size=3, overlap=0)  # step 3

    assert [w["start_token_index"] for w in windows] == [0, 3, 6]
    assert windows[-1]["end_token_index"] == 9


def test_empty_chapter_produces_no_windows():
    stream = chapter_stream("1 Nephi", 1, 0)
    assert create_overlapping_windows([stream], window_size=3, overlap=1) == []


@pytest.mark.parametrize(
    "window_size,overlap",
    [(0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)],
)
def test_invalid_parameters_are_rejected(window_size, overlap):
    stream = chapter_stream("1 Nephi", 1, 10)
    with pytest.raises(ValueError):
        create_overlapping_windows([stream], window_size=window_size, overlap=overlap)


def test_coverage_holds_across_a_real_corpus_if_present():
    """End-to-end: every token of every real chapter is covered."""
    from pathlib import Path

    from chunking import build_chapter_token_stream
    from preprocess import preprocess_bom

    candidates = [Path(__file__).parent / "data" / "bom.txt",
                  Path(__file__).parent / "bom.txt"]
    corpus = next((p for p in candidates if p.exists()), None)
    if corpus is None:
        pytest.skip("no local corpus to check against")

    streams = build_chapter_token_stream(preprocess_bom(str(corpus)))
    windows = create_overlapping_windows(streams, window_size=30, overlap=15)

    covered = {}
    for window in windows:
        key = (window["book"], window["chapter"])
        covered.setdefault(key, set()).update(
            range(window["start_token_index"], window["end_token_index"])
        )

    for stream in streams:
        key = (stream["book"], stream["chapter"])
        assert covered.get(key, set()) == set(range(len(stream["tokens"]))), key
