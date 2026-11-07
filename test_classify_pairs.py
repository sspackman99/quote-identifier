# test_classify_pairs.py
"""The walk over the pair list: orientation, collapsing, resume, output rows."""

import json

import pandas as pd
import pytest

import classify_pairs
from classify_pairs import (
    OUTPUT_COLUMNS,
    build_pair_case,
    collapse_span_pairs,
    key_text,
    load_verse_texts,
    orient,
    pair_key,
    read_journal,
    result_row,
    select,
    write_output,
)

PAIR_COLUMNS = [
    "cosine_similarity", "rank_1", "rank_2",
    "corpus_1", "book_id_1", "book_1", "chapter_1", "verse_1", "end_verse_1",
    "ref_1", "text_1",
    "corpus_2", "book_id_2", "book_2", "chapter_2", "verse_2", "end_verse_2",
    "ref_2", "text_2",
]


def _row(side_1, side_2, cosine, rank_1=1, rank_2=1, text_1="a", text_2="b",
         corpus_1="bom", corpus_2="bom"):
    book_1, chapter_1, verse_1, end_1 = side_1
    book_2, chapter_2, verse_2, end_2 = side_2
    return {
        "cosine_similarity": cosine, "rank_1": rank_1, "rank_2": rank_2,
        "corpus_1": corpus_1, "book_id_1": book_1.lower(), "book_1": book_1,
        "chapter_1": chapter_1, "verse_1": verse_1, "end_verse_1": end_1,
        "ref_1": f"{book_1} {chapter_1}:{verse_1}", "text_1": text_1,
        "corpus_2": corpus_2, "book_id_2": book_2.lower(), "book_2": book_2,
        "chapter_2": chapter_2, "verse_2": verse_2, "end_verse_2": end_2,
        "ref_2": f"{book_2} {chapter_2}:{verse_2}", "text_2": text_2,
    }


def _frame(rows):
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


ORDER = {"1 Nephi": 0, "2 Nephi": 1, "Jacob": 2}


def test_earlier_side_becomes_the_source():
    """Within a canon the later text is the one quoting, so it is the query."""
    oriented = orient(_row(("2 Nephi", 6, 6, 6), ("1 Nephi", 21, 22, 22), 0.99), {"bom": ORDER})

    assert oriented["source_book"] == "1 Nephi"
    assert oriented["source_chapter"] == 21
    assert oriented["query_book"] == "2 Nephi"
    assert oriented["query_chapter"] == 6


def test_orientation_does_not_depend_on_which_side_is_column_1():
    forward = orient(_row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.9), {"bom": ORDER})
    reverse = orient(_row(("2 Nephi", 6, 6, 6), ("1 Nephi", 21, 22, 22), 0.9), {"bom": ORDER})

    assert pair_key(forward) == pair_key(reverse)
    assert key_text(forward) == key_text(reverse)
    assert key_text(forward) == "2 Nephi 6:6 -> 1 Nephi 21:22"


def test_ranks_travel_with_their_side():
    oriented = orient(_row(("2 Nephi", 6, 6, 6), ("1 Nephi", 21, 22, 22), 0.9,
                           rank_1=3, rank_2=1), {"bom": ORDER})

    assert oriented["query_rank"] == 3   # 2 Nephi's own rank for this pair
    assert oriented["source_rank"] == 1


def test_collapse_keeps_one_entry_per_span_pair_and_all_windows():
    frame = _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.95),
        _row(("2 Nephi", 6, 6, 6), ("1 Nephi", 21, 22, 22), 0.99),
        _row(("1 Nephi", 21, 26, 26), ("2 Nephi", 6, 18, 18), 0.98),
    ])

    entries = collapse_span_pairs(frame, {"bom": ORDER})

    assert len(entries) == 2, "three rows describe two passage pairs"
    by_score = entries[0]
    assert by_score["cosine_similarity"] == pytest.approx(0.99)
    assert len(by_score["windows"]) == 3, "every supporting window is kept as evidence"
    assert [entry["cosine_similarity"] for entry in entries] == sorted(
        [entry["cosine_similarity"] for entry in entries], reverse=True
    )


def test_select_applies_the_cutoff_before_the_limit():
    frame = _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.95),
        _row(("1 Nephi", 21, 26, 26), ("2 Nephi", 6, 18, 18), 0.85),
        _row(("1 Nephi", 21, 25, 25), ("2 Nephi", 6, 17, 17), 0.60),
    ])
    entries = collapse_span_pairs(frame, {"bom": ORDER})

    assert len(select(entries, 0.80, 0)) == 2
    assert len(select(entries, 0.80, 1)) == 1
    assert select(entries, 0.80, 1)[0]["cosine_similarity"] == pytest.approx(0.95)


@pytest.fixture(scope="module")
def bom_texts():
    return load_verse_texts(("bom", "data/bom.txt"))


def _entry(frame):
    return collapse_span_pairs(frame, {"bom": ORDER})[0]


def test_case_carries_both_passages_their_context_and_the_evidence(bom_texts):
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))

    case = build_pair_case(entry, bom_texts)

    assert case["query"]["ref"] == "2 Nephi 6:6"
    assert case["source"]["ref"] == "1 Nephi 21:22"
    assert "mine hand" in case["source"]["text"].lower()
    assert case["query"]["context"], "context must be supplied on both sides"
    assert case["source"]["context"]
    assert case["matched_windows"][0]["similarity"] == pytest.approx(0.99)
    assert case["retrieval"]["best"] == pytest.approx(0.99)
    assert case["reproduction"]["verses_in_source"] == 1


def test_case_measures_reproduction_from_the_wording(bom_texts):
    """Isaiah 49 wording reproduced in 2 Nephi 6 must measure as reproduced."""
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))

    case = build_pair_case(entry, bom_texts)

    assert case["reproduction"]["reproduced_count"] == 1
    assert case["reproduction"]["reproduced_share"] == pytest.approx(1.0)
    assert case["reproduction"]["extent"] in ("quotation", "partial_quotation")


def test_result_row_flattens_the_classification(bom_texts):
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))
    result = {
        "relationship": "quotation",
        "secondary_relationship": "",
        "confidence": 0.92,
        "needs_review": False,
        "review_reasons": [],
        "query_span": "6",
        "source_span": "22",
        "guard_applied": "",
        "guard_original_relationship": "",
        "reproduction": {"reproduced_share": 1.0, "extent": "quotation"},
        "explanation": "2 Nephi 6:6 reproduces 1 Nephi 21:22.",
        "evidence": ["mine hand", "my standard"],
        "model": "fake-model",
        "rubric_version": "1.0",
    }

    row = result_row(entry, result)

    assert list(row) == OUTPUT_COLUMNS or set(row) == set(OUTPUT_COLUMNS)
    assert row["query_ref"] == "2 Nephi 6:6"
    assert row["source_ref"] == "1 Nephi 21:22"
    assert row["relationship"] == "quotation"
    assert row["evidence"] == "mine hand | my standard"
    assert row["error"] == ""


def test_error_row_keeps_the_pair_and_the_message():
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))

    row = result_row(entry, error=RuntimeError("endpoint unreachable"))

    assert row["query_ref"] == "2 Nephi 6:6"
    assert row["relationship"] is None
    assert "endpoint unreachable" in row["error"]


def test_error_row_leaves_numeric_columns_blank_not_text(tmp_path):
    """A failed pair must not turn confidence/reproduced_share into strings."""
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))
    records = [{"key": entry["key_text"], "row": result_row(entry, error="boom")}]

    output = tmp_path / "out.csv"
    write_output(records, output)
    frame = pd.read_csv(output)

    assert pd.isna(frame.iloc[0]["confidence"])
    assert pd.isna(frame.iloc[0]["reproduced_share"])
    assert frame.iloc[0]["error"] == "boom"


def test_journal_round_trips_and_skips_a_truncated_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    record = {"key": "2 Nephi 6:6 -> 1 Nephi 21:22", "row": {"cosine_similarity": 0.9}}
    classify_pairs.append_journal(path, record)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"key": "truncated", "row": {')  # killed mid-write

    records = read_journal(path)

    assert list(records) == ["2 Nephi 6:6 -> 1 Nephi 21:22"]
    assert records["2 Nephi 6:6 -> 1 Nephi 21:22"]["row"]["cosine_similarity"] == 0.9


def test_write_output_uses_the_declared_column_order(tmp_path):
    entry = _entry(_frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]))
    records = [{"key": entry["key_text"], "row": result_row(entry, error="boom")}]

    output = tmp_path / "out.csv"
    frame = write_output(records, output)

    assert list(frame.columns) == OUTPUT_COLUMNS
    assert output.exists()
    assert len(pd.read_csv(output)) == 1


ANSWER = {
    "relationship": "quotation",
    "confidence": 0.9,
    "query_span": "6",
    "source_span": "22",
    "secondary_relationship": "",
    "explanation": "2 Nephi reproduces the Isaiah wording.",
    "evidence": ["mine hand"],
    "needs_review": False,
}


def test_main_walks_selected_pairs_and_writes_csv(tmp_path, monkeypatch, bom_texts, capsys):
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
        _row(("1 Nephi", 21, 26, 26), ("2 Nephi", 6, 18, 18), 0.50),  # below cutoff
    ]).to_csv(pairs, index=False)

    calls = []

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            calls.append(user_prompt)
            return json.dumps(ANSWER)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)

    status = classify_pairs.main([
        "--input", str(pairs),
        "--output", str(tmp_path / "out.csv"),
        "--journal", str(tmp_path / "out.jsonl"),
        "--corpus", "data/bom.txt",
        "--min-similarity", "0.80",
    ])

    out = pd.read_csv(tmp_path / "out.csv")
    assert status == 0
    assert len(calls) == 1, "the pair below the cutoff must not cost a model call"
    assert len(out) == 1
    assert out.iloc[0]["relationship"] == "quotation"
    assert out.iloc[0]["query_ref"] == "2 Nephi 6:6"
    assert "model calls to make: 1" in capsys.readouterr().out


def test_main_resumes_without_reclassifying(tmp_path, monkeypatch, bom_texts):
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]).to_csv(pairs, index=False)

    calls = []

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            calls.append(user_prompt)
            return json.dumps(ANSWER)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)
    argv = ["--input", str(pairs), "--output", str(tmp_path / "out.csv"),
            "--journal", str(tmp_path / "out.jsonl"), "--corpus", "data/bom.txt"]

    classify_pairs.main(argv)
    classify_pairs.main(argv)

    assert len(calls) == 1, "the journal must prevent a second paid call"
    assert len(pd.read_csv(tmp_path / "out.csv")) == 1


def test_main_records_an_error_without_stopping(tmp_path, monkeypatch, bom_texts, capsys):
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
        _row(("1 Nephi", 21, 26, 26), ("2 Nephi", 6, 18, 18), 0.98),
    ]).to_csv(pairs, index=False)

    attempts = []

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            attempts.append(user_prompt)
            if len(attempts) == 1:
                raise RuntimeError("classifier endpoint unreachable")
            return json.dumps(ANSWER)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)

    status = classify_pairs.main([
        "--input", str(pairs), "--output", str(tmp_path / "out.csv"),
        "--journal", str(tmp_path / "out.jsonl"), "--corpus", "data/bom.txt",
    ])

    out = pd.read_csv(tmp_path / "out.csv")
    assert status == 0
    assert len(attempts) == 2, "one failure must not end the walk"
    assert out["error"].fillna("").str.contains("unreachable").sum() == 1
    assert (out["relationship"] == "quotation").sum() == 1


def test_malformed_answer_is_repaired_with_one_extra_call(tmp_path, monkeypatch, bom_texts):
    """A broken JSON answer costs one repair call, not a lost pair."""
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]).to_csv(pairs, index=False)

    prompts = []
    answers = [
        '{"relationship": "quotation", "confidence": 0.9',   # unparseable
        json.dumps(ANSWER),
    ]

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            prompts.append(user_prompt)
            return answers.pop(0)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)

    status = classify_pairs.main([
        "--input", str(pairs), "--output", str(tmp_path / "out.csv"),
        "--journal", str(tmp_path / "out.jsonl"), "--corpus", "data/bom.txt",
    ])

    out = pd.read_csv(tmp_path / "out.csv")
    assert status == 0
    assert len(prompts) == 2
    assert "YOUR PREVIOUS ANSWER WAS REJECTED" in prompts[1]
    assert out.iloc[0]["relationship"] == "quotation"
    assert int(out.iloc[0]["attempts"]) == 2
    assert pd.isna(out.iloc[0]["error"])


def test_resume_retries_a_row_that_failed(tmp_path, monkeypatch, bom_texts):
    """A journaled error is not a result: the next run must try it again."""
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.99),
    ]).to_csv(pairs, index=False)

    calls = {"count": 0}

    def failing_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            calls["count"] += 1
            raise RuntimeError("classifier endpoint unreachable")
        return complete

    def working_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            calls["count"] += 1
            return json.dumps(ANSWER)
        return complete

    argv = ["--input", str(pairs), "--output", str(tmp_path / "out.csv"),
            "--journal", str(tmp_path / "out.jsonl"), "--corpus", "data/bom.txt"]

    monkeypatch.setattr(classify_pairs, "http_completer", failing_completer)
    classify_pairs.main(argv)
    first = pd.read_csv(tmp_path / "out.csv")
    assert first.iloc[0]["error"].startswith("classifier endpoint unreachable")

    monkeypatch.setattr(classify_pairs, "http_completer", working_completer)
    classify_pairs.main(argv)

    out = pd.read_csv(tmp_path / "out.csv")
    assert calls["count"] == 2, "the failed pair must be attempted again"
    assert out.iloc[0]["relationship"] == "quotation"
    assert pd.isna(out.iloc[0]["error"])


def test_cross_corpus_pair_takes_its_source_from_the_named_corpus():
    """Bible text is always the source when comparing Book of Mormon to KJV."""
    row = _row(("Mosiah", 14, 1, 1), ("Isaiah", 53, 1, 1), 0.97,
               corpus_1="bom", corpus_2="kjv")

    oriented = orient(row, {"bom": {"Mosiah": 0}, "kjv": {"Isaiah": 0}},
                      source_corpus="kjv")

    assert oriented["source_corpus"] == "kjv"
    assert oriented["source_book"] == "Isaiah"
    assert oriented["query_corpus"] == "bom"
    assert oriented["query_book"] == "Mosiah"


def test_cross_corpus_key_names_both_corpora():
    row = _row(("Mosiah", 14, 1, 1), ("Isaiah", 53, 1, 1), 0.97,
               corpus_1="bom", corpus_2="kjv")

    oriented = orient(row, {"bom": {"Mosiah": 0}, "kjv": {"Isaiah": 0}},
                      source_corpus="kjv")

    assert key_text(oriented) == "Mosiah 14:1 [bom] -> Isaiah 53:1 [kjv]"


def test_within_corpus_key_is_unchanged_so_old_journals_still_resume():
    oriented = orient(_row(("1 Nephi", 21, 22, 22), ("2 Nephi", 6, 6, 6), 0.9),
                      {"bom": ORDER})

    assert key_text(oriented) == "2 Nephi 6:6 -> 1 Nephi 21:22"


def test_cross_corpus_case_reads_each_side_from_its_own_corpus():
    texts = load_verse_texts(("bom", "data/bom.txt"), ("kjv", "data/kjv.txt"))
    row = _row(("Mosiah", 14, 1, 1), ("Isaiah", 53, 1, 1), 0.97,
               corpus_1="bom", corpus_2="kjv")
    entry = collapse_span_pairs(_frame([row]),
                                {"bom": {"Mosiah": 0}, "kjv": {"Isaiah": 0}},
                                source_corpus="kjv")[0]

    case = build_pair_case(entry, texts)

    assert case["query"]["ref"] == "Mosiah 14:1"
    assert case["source"]["ref"] == "Isaiah 53:1"
    assert "who hath believed" in case["source"]["text"].lower()
    assert "who hath believed" in case["query"]["text"].lower()
    assert case["reproduction"]["reproduced_count"] == 1


def test_main_classifies_a_cross_corpus_pair(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("Mosiah", 14, 1, 1), ("Isaiah", 53, 1, 1), 0.97,
             corpus_1="bom", corpus_2="kjv"),
    ]).to_csv(pairs, index=False)

    def fake_completer(**kwargs):
        def complete(system_prompt, user_prompt):
            return json.dumps(ANSWER)
        return complete

    monkeypatch.setattr(classify_pairs, "http_completer", fake_completer)

    status = classify_pairs.main([
        "--input", str(pairs), "--output", str(tmp_path / "out.csv"),
        "--journal", str(tmp_path / "out.jsonl"),
        "--corpus", "data/bom.txt", "--corpus-id", "bom",
        "--source-corpus", "data/kjv.txt", "--source-corpus-id", "kjv",
    ])

    out = pd.read_csv(tmp_path / "out.csv")
    assert status == 0
    assert out.iloc[0]["query_corpus"] == "bom"
    assert out.iloc[0]["source_corpus"] == "kjv"
    assert out.iloc[0]["query_ref"] == "Mosiah 14:1"
    assert out.iloc[0]["source_ref"] == "Isaiah 53:1"


def test_main_refuses_a_mixed_pair_list_without_a_named_source(tmp_path, monkeypatch):
    pairs = tmp_path / "pairs.csv"
    _frame([
        _row(("Mosiah", 14, 1, 1), ("Isaiah", 53, 1, 1), 0.97,
             corpus_1="bom", corpus_2="kjv"),
    ]).to_csv(pairs, index=False)

    with pytest.raises(SystemExit) as exit_info:
        classify_pairs.main([
            "--input", str(pairs), "--output", str(tmp_path / "out.csv"),
            "--journal", str(tmp_path / "out.jsonl"),
        ])

    assert exit_info.value.code == 2
