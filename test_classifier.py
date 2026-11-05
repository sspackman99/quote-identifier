# test_classifier.py
"""Tests for the classification stage, with the model mocked out.

The point of these tests is the contract: what the classifier is given, what it is
allowed to return, and when a human has to look. No network, no model.
"""

import json

import pytest

from classifier import (
    RELATIONSHIPS,
    ClassifierResponseError,
    RelationshipClassifier,
    apply_review_policy,
    build_case,
    http_completer,
    parse_response,
    render_prompt,
)

CASE = {
    "query": {"ref": "Mosiah 12", "span": "21-24",
              "text": "how beautiful upon the mountains are the feet of him that "
                      "bringeth good tidings",
              "context": "the preceding verses about the watchmen"},
    "source": {"ref": "Isaiah 52", "span": "7-10",
               "text": "how beautiful upon the mountains are the feet of him that "
                       "bringeth good tidings",
               "context": "the preceding verses about Jerusalem"},
    "matched_windows": [{"query_range": "21-24", "source_range": "7-9", "similarity": 0.93}],
    "retrieval": {"best": 0.93, "support": 2, "coverage": 0.4},
}


def response(**overrides):
    payload = {
        "relationship": "partial_quotation",
        "confidence": 0.94,
        "query_span": "21-24",
        "source_span": "7-10",
        "explanation": "The wording closely follows the source passage.",
        "evidence": ["how beautiful upon the mountains are the feet of him"],
        "needs_review": False,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_build_case_requires_matched_evidence():
    with pytest.raises(ValueError):
        build_case({"ref": "a", "span": "1", "text": "t"},
                   {"ref": "b", "span": "1", "text": "t"}, [])


def test_build_case_defaults_context_to_empty_rather_than_missing():
    case = build_case({"ref": "a", "span": "1", "text": "t"},
                      {"ref": "b", "span": "1", "text": "t"},
                      [{"query_range": "1", "source_range": "1", "similarity": 0.9}])

    assert case["query"]["context"] == ""
    assert case["source"]["context"] == ""


def test_prompt_keeps_matched_evidence_and_context_apart():
    prompt = render_prompt(build_case(CASE["query"], CASE["source"],
                                      CASE["matched_windows"], CASE["retrieval"]))

    matched_at = prompt.index("MATCHED EVIDENCE")
    context_at = prompt.index("SURROUNDING CONTEXT OF A")

    assert context_at < matched_at, "context should be presented before the match"
    assert "query 21-24  <->  source 7-9" in prompt
    assert "similarity 0.9300" in prompt
    assert "not matched evidence" in prompt
    assert prompt.count("SURROUNDING CONTEXT") == 2


def test_instructions_states_that_similarity_is_not_dependence():
    from classifier import SYSTEM_PROMPT

    assert "not evidence of dependence" in SYSTEM_PROMPT
    assert "needs_review to true" in SYSTEM_PROMPT


def test_parse_accepts_a_plain_json_object():
    result = parse_response(response())

    assert result["relationship"] == "partial_quotation"
    assert result["confidence"] == 0.94
    assert result["evidence"] == ["how beautiful upon the mountains are the feet of him"]
    assert result["model_needs_review"] is False


def test_parse_accepts_a_fenced_json_block():
    assert parse_response("```json\n" + response() + "\n```")["relationship"] == "partial_quotation"


def test_parse_tolerates_prose_around_the_object():
    assert parse_response("Here is my answer:\n" + response() + "\nHope that helps.")[
        "relationship"] == "partial_quotation"


def test_parse_rejects_a_label_outside_the_taxonomy():
    with pytest.raises(ClassifierResponseError) as error:
        parse_response(response(relationship="borrowing"))

    assert "taxonomy" in str(error.value)


def test_parse_normalises_spacing_in_the_label():
    assert parse_response(response(relationship="Partial Quotation"))[
        "relationship"] == "partial_quotation"


def test_parse_rejects_confidence_outside_the_unit_interval():
    with pytest.raises(ClassifierResponseError):
        parse_response(response(confidence=1.4))
    with pytest.raises(ClassifierResponseError):
        parse_response(response(confidence=-0.2))


def test_parse_rejects_a_non_numeric_confidence():
    with pytest.raises(ClassifierResponseError):
        parse_response(response(confidence="high"))


def test_parse_rejects_a_dependent_label_with_no_evidence():
    """A quotation claim has to quote something."""
    with pytest.raises(ClassifierResponseError) as error:
        parse_response(response(evidence=[]))

    assert "without any evidence" in str(error.value)


def test_parse_allows_unrelated_with_no_evidence():
    result = parse_response(response(relationship="unrelated", evidence=[],
                                     query_span="", source_span="",
                                     explanation="Different subjects entirely."))

    assert result["relationship"] == "unrelated"


def test_parse_coerces_a_single_evidence_string_into_a_list():
    assert parse_response(response(evidence="one phrase"))["evidence"] == ["one phrase"]


def test_parse_rejects_an_empty_explanation():
    with pytest.raises(ClassifierResponseError):
        parse_response(response(explanation="   "))


def test_parse_rejects_a_response_with_no_json_at_all():
    with pytest.raises(ClassifierResponseError):
        parse_response("I think these are related.")


def test_high_confidence_with_evidence_and_spans_needs_no_review():
    result = apply_review_policy(parse_response(response()), CASE)

    assert result["needs_review"] is False
    assert result["review_reasons"] == []


def test_low_confidence_escalates_to_review():
    result = apply_review_policy(parse_response(response(confidence=0.55)), CASE)

    assert result["needs_review"] is True
    assert any("below" in reason for reason in result["review_reasons"])


def test_a_model_flagged_case_escalates_even_when_confident():
    result = apply_review_policy(parse_response(response(needs_review=True)), CASE)

    assert result["needs_review"] is True
    assert "model flagged the case" in result["review_reasons"]


def test_unrelated_at_high_similarity_is_escalated():
    """Contradicting strong evidence is exactly what a human should look at."""
    case = dict(CASE, retrieval={"best": 0.95})
    result = apply_review_policy(
        parse_response(response(relationship="unrelated", evidence=[], confidence=0.9,
                                explanation="Coincidental shared phrasing.")),
        case,
    )

    assert result["needs_review"] is True
    assert any("unrelated at similarity" in reason for reason in result["review_reasons"])


def test_dependence_without_spans_is_escalated():
    result = apply_review_policy(
        parse_response(response(query_span="", source_span="")), CASE)

    assert result["needs_review"] is True
    assert any("without verse spans" in reason for reason in result["review_reasons"])


def test_classifier_round_trip_records_where_the_evidence_came_from():
    seen = {}

    def fake_complete(system_prompt, user_prompt):
        seen["system"] = system_prompt
        seen["user"] = user_prompt
        return response()

    classifier = RelationshipClassifier(fake_complete, model="fake-model")
    result = classifier.classify(CASE)

    assert result["relationship"] == "partial_quotation"
    assert result["model"] == "fake-model"
    assert result["evidence_source"] == "matched"
    assert result["context_supplied"] is True
    assert result["matched_windows"] == CASE["matched_windows"]
    assert "PASSAGE A" in seen["user"] and "PASSAGE B" in seen["user"]


def test_classifier_marks_a_case_without_context_as_such():
    bare = build_case(CASE["query"] | {"context": ""}, CASE["source"] | {"context": ""},
                      CASE["matched_windows"], CASE["retrieval"])
    result = RelationshipClassifier(lambda system, user: response()).classify(bare)

    assert result["context_supplied"] is False


def test_classifier_rejects_a_non_callable_provider():
    with pytest.raises(TypeError):
        RelationshipClassifier("not a function")


def test_provider_failure_is_visible_not_swallowed():
    def broken(system_prompt, user_prompt):
        raise RuntimeError("endpoint unreachable")

    with pytest.raises(RuntimeError):
        RelationshipClassifier(broken).classify(CASE)


def test_http_completer_requires_an_endpoint(monkeypatch):
    monkeypatch.delenv("QI_LLM_URL", raising=False)

    with pytest.raises(ValueError):
        http_completer()


def test_the_taxonomy_matches_the_evaluation_labels():
    """Classifier labels and evaluator labels must stay the same vocabulary."""
    from evaluation_pairs import ALL_KINDS

    assert set(RELATIONSHIPS) == set(ALL_KINDS)


class _FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_completer_sends_a_bounded_non_thinking_request(monkeypatch):
    """Classification is a short answer; do not let a reasoning model ramble."""
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _FakeHTTPResponse({"choices": [{"message": {"content": response()}}]})

    monkeypatch.setattr("classifier.urllib.request.urlopen", fake_urlopen)
    complete = http_completer(url="http://example/v1/chat/completions",
                              model="m", timeout=11)

    complete("system", "user")

    assert captured["body"]["max_tokens"] == 700
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert captured["body"]["temperature"] == 0.0
    assert captured["timeout"] == 11
    assert len(captured["body"]["messages"]) == 2


def test_http_completer_can_enable_thinking_when_asked(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["body"] = json.loads(request.data)
        return _FakeHTTPResponse({"choices": [{"message": {"content": response()}}]})

    monkeypatch.setattr("classifier.urllib.request.urlopen", fake_urlopen)
    http_completer(url="http://example/v1/chat/completions", model="m",
                   thinking=True)("system", "user")

    assert "chat_template_kwargs" not in captured["body"]
