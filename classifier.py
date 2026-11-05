"""Classify the relationship between a retrieved candidate and its query passage.

Retrieval answers "these two passages look alike". This module answers "what kind of
relationship is that, and how sure are we". Similarity is EVIDENCE for the decision,
never the decision itself, so the model sees the matched evidence and the surrounding
context as two clearly separated things and is told which one it may cite as support
for a quotation.

The module is deliberately provider-agnostic and does no I/O of its own:

    case = build_case(query=..., source=..., matched_windows=..., ...)
    classifier = RelationshipClassifier(complete)
    result = classifier.classify(case)

`complete` is any callable (system_prompt, user_prompt) -> str. `http_completer()`
provides one for an OpenAI-compatible chat endpoint. Everything the model returns is
parsed and validated here; a malformed or out-of-taxonomy answer is an error, not a
silent 'unrelated'.
"""

import json
import os
import re
import urllib.error
import urllib.request

RELATIONSHIPS = (
    "quotation",
    "partial_quotation",
    "paraphrase",
    "allusion",
    "interpretation",
    "shared_vocabulary",
    "unrelated",
)

# Relationships that assert textual dependence; the rest do not.
DEPENDENT = ("quotation", "partial_quotation", "paraphrase", "allusion", "interpretation")

RUBRIC_VERSION = "1.0"
MATCHED_WINDOW_LIMIT = 8
DEFAULT_MODEL = "local-classifier"
DEFAULT_TIMEOUT = 180  # this model can take a while to think
REVIEW_CONFIDENCE_FLOOR = 0.75
COVERAGE_FOR_FULL_QUOTATION = 0.60  # same floor the rubric uses
WHOLE_PASSAGE_SHARE = 0.60
NO_REPRODUCTION_SHARE = 0.10
CONTRADICTION_SIMILARITY = 0.85

SYSTEM_PROMPT = """You classify the relationship between two scriptural passages.

Rubric version 1.0. Answer with ONE of these labels:

- quotation: the query reproduces a substantial, continuous portion of the source, with
  wording substantially preserved. Key test: is it essentially a direct quotation?
- partial_quotation: a shorter, continuous, substantially verbatim excerpt from a larger
  source passage. Key test: is the wording directly reproduced, but only a portion?
- paraphrase: the source's substantive proposition is restated in different wording,
  without merely explaining its implications. Key test: same assertion, other words?
- allusion: an identifiable reference through distinctive language, imagery, structure or a
  recognizable combination of ideas, without reproducing or restating enough to be a
  quotation or paraphrase. Key test: is the source being evoked rather than restated?
- interpretation: the passage explains, applies, expands, draws an inference from, or
  develops the meaning of another passage. Key test: is it explaining what the source means
  or implies, rather than restating what it says?
- shared_vocabulary: similar words or conventional expressions, with no persuasive evidence
  of a specific textual relationship. Key test: could this similarity occur independently?
- unrelated: no meaningful textual relationship is supported.

Boundaries that decide most cases:

1. Quotation versus partial_quotation is about the EXTENT of the reproduced text, not its
   importance. A short but exact quotation is partial_quotation even when it is the most
   memorable line in the source. There is no verse-count rule: one verse can carry a
   substantial quotation, several verses can carry only a short excerpt. Judge the
   continuous wording actually reproduced. Example: two verses lifted exactly out of a long
   chapter = partial_quotation; the whole servant song reproduced in order = quotation.
2. Paraphrase versus interpretation is about the RELATION between the statements.
   Paraphrase says substantially the same thing in different words. Interpretation
   explains, applies or draws conclusions from it. Example: a description of charity
   reworded = paraphrase; an explanation of who the servant in the source is, quoting a
   phrase or two = interpretation.
3. Allusion versus shared_vocabulary: an allusion evokes a specific passage through
   distinctive imagery or structure; shared_vocabulary is language that any text in this
   tradition could carry independently. Example: a plea to turn away divine anger modelled
   on a specific psalm = allusion; a general legal principle about individual
   responsibility = shared_vocabulary.
4. A passage may quote inside an explanation. If the explanation dominates, answer
   interpretation and record the embedded quotation as secondary; if the quoted block
   dominates, answer quotation or partial_quotation and record the comment as secondary.
5. A passage may relate to more than one source. Judge the pair you are given, against its
   own verses, and do not let a second relationship dilute the answer.

Two tests to apply before answering, because they decide the cases that go wrong:

EXTENT TEST (before quotation or partial_quotation).
Count how much of the CITED SOURCE passage the query reproduces, and compare it with the
size of that source passage.
  - reproduces most of the cited source passage -> quotation
  - reproduces only part of it, even a long or memorable part -> partial_quotation
Coverage of the source decides, not the amount of text in the query. A query can be long
and still quote only an excerpt; a query can be short and still reproduce a whole short
psalm or paragraph.

DOMINANCE TEST (before interpretation).
If the passage both quotes and explains, compare how much of the query is reproduction
and how much is explanation. The larger share decides the label; the other relationship
goes in secondary_relationship. Exposition wrapped around one quoted clause is
interpretation; a quoted block with a sentence of comment is a quotation.

How to decide:
1. The MATCHED evidence below is the only thing that can support quotation or
   partial_quotation. Quote from it when you claim verbatim reuse.
2. SURROUNDING CONTEXT is provided to help you understand both passages. Similarity that
   exists only in the surrounding context is never a quotation.
3. A high similarity score is not by itself evidence of dependence.
4. When two relationships both apply, choose the dominant one and name the secondary one
   in the explanation. Do not force artificial certainty.
5. If the evidence is genuinely ambiguous, still choose the best label AND set
   needs_review to true. Do not inflate confidence to seem decisive.

Return JSON only, with no prose around it:

{
  "relationship": "<one label above>",
  "confidence": <0.0 to 1.0>,
  "query_span": "<verses in the query carrying the relationship>",
  "source_span": "<verses in the source being used>",
  "secondary_relationship": "<another label, or \"\" if none>",
  "explanation": "<one or two sentences>",
  "evidence": ["<short quoted phrases from the MATCHED evidence>", "..."],
  "needs_review": <true or false>
}

If the relationship is "unrelated", evidence may be empty and the spans may be "".
"""


class ClassifierResponseError(ValueError):
    """The model returned something that cannot be used as a classification."""


def build_case(query, source, matched_windows, retrieval=None, context_radius=2,
               reproduction=None):
    """Assemble the classifier input, keeping matched evidence and context apart.

    query / source: {"ref": "Mosiah 12", "span": "21-24", "text": "..."}
    matched_windows: [{"query_range", "source_range", "similarity"}, ...]
    """
    if not matched_windows:
        raise ValueError("a case needs at least one matched window")

    ordered = sorted((dict(window) for window in matched_windows),
                     key=lambda window: -window["similarity"])
    kept = ordered[:MATCHED_WINDOW_LIMIT]

    case = {
        "query": dict(query),
        "source": dict(source),
        "matched_windows": kept,
        "matched_total": len(ordered),
        "matched_truncated": len(ordered) > len(kept),
        "retrieval": dict(retrieval or {}),
        "reproduction": dict(reproduction or {}),
        "context_radius": context_radius,
    }
    case["query"].setdefault("context", "")
    case["source"].setdefault("context", "")
    return case


def render_prompt(case):
    """Render the case as a prompt. Matched and contextual text never merge."""
    query = case["query"]
    source = case["source"]
    retrieval = case.get("retrieval", {})

    matched_lines = []
    for window in case["matched_windows"]:
        matched_lines.append(
            f"  query {window['query_range']}  <->  source {window['source_range']}"
            f"   similarity {window['similarity']:.4f}"
        )

    similarity_note = []
    if retrieval.get("best") is not None:
        similarity_note.append(f"best similarity {retrieval['best']:.4f}")
    if retrieval.get("support") is not None:
        similarity_note.append(f"{retrieval['support']} distinct supporting ranges")
    if retrieval.get("coverage") is not None:
        similarity_note.append(f"coverage of the candidate passage {retrieval['coverage']:.0%}")

    sections = [
        "PASSAGE A (the query: the text that may be quoting)",
        f"reference: {query['ref']}",
        f"verses in question: {query['span']}",
        query["text"].strip(),
    ]

    if query.get("context"):
        sections += ["", "SURROUNDING CONTEXT OF A (not matched evidence)", query["context"].strip()]

    sections += [
        "",
        "PASSAGE B (the candidate source)",
        f"reference: {source['ref']}",
        f"verses in question: {source['span']}",
        source["text"].strip(),
    ]

    if source.get("context"):
        sections += ["", "SURROUNDING CONTEXT OF B (not matched evidence)", source["context"].strip()]

    reproduction = case.get("reproduction") or {}
    if reproduction.get("reproduced_share") is not None:
        sections += [
            "",
            "MEASURED REPRODUCTION (computed from the text, not a model judgement)",
            f"  source verses reproduced: {reproduction.get('reproduced_count')} of "
            f"{reproduction.get('verses_in_source')} "
            f"({reproduction['reproduced_share']:.0%} of the source passage)",
            f"  longest run of consecutive reproduced verses: {reproduction.get('longest_run')}",
            f"  share of the query's wording drawn from the source: "
            f"{reproduction.get('query_share', 0.0):.0%}",
            "  Apply the extent test to this measurement: most of the passage reproduced "
            "means quotation; a run inside a longer passage means partial_quotation.",
        ]

    total = case.get("matched_total", len(case["matched_windows"]))
    heading = "MATCHED EVIDENCE (retrieval's window pairs"
    if case.get("matched_truncated"):
        heading += f"; the {len(case['matched_windows'])} strongest of {total}"
    heading += ")"
    sections += ["", heading]
    sections += matched_lines
    if similarity_note:
        sections += ["", "retrieval summary: " + "; ".join(similarity_note)]

    sections += [
        "",
        "Classify the relationship of A to B and return the JSON object.",
    ]
    return "\n".join(sections)


def _strip_code_fence(text):
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def _extract_object(text):
    """Find the first balanced JSON object, tolerating prose around it."""
    cleaned = _strip_code_fence(text)
    start = cleaned.find("{")
    if start == -1:
        raise ClassifierResponseError(f"no JSON object in response: {text[:200]!r}")

    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(cleaned)):
        character = cleaned[position]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:position + 1]

    raise ClassifierResponseError(f"unbalanced JSON in response: {text[:200]!r}")


def parse_response(text):
    """Parse and validate one model response into a structured result."""
    payload = json.loads(_extract_object(text))

    relationship = str(payload.get("relationship", "")).strip().lower().replace(" ", "_")
    if relationship not in RELATIONSHIPS:
        raise ClassifierResponseError(
            f"relationship {payload.get('relationship')!r} is not in the taxonomy"
        )

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        raise ClassifierResponseError(
            f"confidence {payload.get('confidence')!r} is not a number"
        )
    if not 0.0 <= confidence <= 1.0:
        raise ClassifierResponseError(f"confidence {confidence} is outside 0..1")

    evidence = payload.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    evidence = [str(item).strip() for item in evidence if str(item).strip()]

    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        raise ClassifierResponseError("explanation is empty")

    if relationship != "unrelated" and not evidence:
        raise ClassifierResponseError(
            f"{relationship} was claimed without any evidence quoted from the match"
        )

    secondary = str(payload.get("secondary_relationship", "") or "").strip().lower().replace(" ", "_")
    if secondary and secondary not in RELATIONSHIPS:
        raise ClassifierResponseError(
            f"secondary_relationship {secondary!r} is not in the taxonomy"
        )
    if secondary == relationship:
        secondary = ""

    return {
        "relationship": relationship,
        "secondary_relationship": secondary,
        "confidence": round(confidence, 4),
        "query_span": str(payload.get("query_span", "")).strip(),
        "source_span": str(payload.get("source_span", "")).strip(),
        "explanation": explanation,
        "evidence": evidence,
        "model_needs_review": bool(payload.get("needs_review", False)),
    }


def apply_extent_guard(result, case):
    """Enforce the rubric's extent rule when the evidence can measure it.

    The rubric says a quotation reproduces the source substantially, and an excerpt
    is a partial quotation however long or memorable. A small model keeps calling
    excerpts quotations even when told the rule, so where retrieval can measure how
    much of the cited source the match covers, that measurement decides - downwards
    only. The guard never promotes a label.
    """
    reproduction = case.get("reproduction") or {}
    share = reproduction.get("reproduced_share")
    result["coverage_of_source"] = case.get("retrieval", {}).get("coverage")
    result["reproduced_share"] = share
    result["guard_applied"] = ""

    if share is None:
        return result

    if result["relationship"] == "quotation" and share < NO_REPRODUCTION_SHARE:
        # claiming a quotation where almost nothing is reproduced is the worst error
        # in this taxonomy: the wording is not there at all
        result["guard_original_relationship"] = result["relationship"]
        result["relationship"] = "paraphrase"
        result["guard_applied"] = "extent-no-reproduction"
        result["guard_note"] = (
            f"measurement reproduces only {share:.0%} of the source passage, so a "
            f"quotation cannot be supported; downgraded to paraphrase for review"
        )
        result["model_needs_review"] = True

    elif result["relationship"] == "quotation" and share < COVERAGE_FOR_FULL_QUOTATION:
        result["guard_original_relationship"] = result["relationship"]
        result["relationship"] = "partial_quotation"
        result["guard_applied"] = "extent"
        result["guard_note"] = (
            f"measurement reproduces {share:.0%} of the source passage, under the "
            f"{COVERAGE_FOR_FULL_QUOTATION:.0%} needed for a whole-passage quotation"
        )
    elif result["relationship"] == "partial_quotation" and share >= WHOLE_PASSAGE_SHARE:
        # never promote automatically, but say what the measurement saw
        result["guard_applied"] = "observation"
        result["guard_note"] = (
            f"measurement reproduces {share:.0%} of the source passage; the rubric's "
            f"extent reading would be quotation - left for review, not promoted"
        )

    return result


def apply_review_policy(result, case):
    """Escalate to human review where the evidence and the label disagree.

    The model is allowed to say needs_review itself; this adds the cases it should
    not be able to talk its way out of.
    """
    reasons = []
    if result["model_needs_review"]:
        reasons.append("model flagged the case")
    if result["confidence"] < REVIEW_CONFIDENCE_FLOOR:
        reasons.append(f"confidence {result['confidence']:.2f} below {REVIEW_CONFIDENCE_FLOOR}")
    if result["relationship"] == "unrelated":
        best = case.get("retrieval", {}).get("best")
        if best is not None and best >= CONTRADICTION_SIMILARITY:
            reasons.append(
                f"called unrelated at similarity {best:.2f}; that needs a human"
            )
    if result["relationship"] in DEPENDENT and not (result["query_span"] or result["source_span"]):
        reasons.append("dependent relationship claimed without verse spans")
    if result.get("guard_applied") and result.get("guard_note"):
        reasons.append(f"guard: {result['guard_note']}")

    result["needs_review"] = bool(reasons)
    result["review_reasons"] = reasons
    return result


class RelationshipClassifier:
    """Thin wrapper: build the prompt, call the provider, validate, apply policy."""

    def __init__(self, complete, model=None):
        if not callable(complete):
            raise TypeError("complete must be a callable (system, user) -> str")
        self._complete = complete
        self.model = model or DEFAULT_MODEL

    def classify(self, case):
        response = self._complete(SYSTEM_PROMPT, render_prompt(case))
        result = parse_response(response)
        result["model"] = self.model
        result["rubric_version"] = RUBRIC_VERSION
        result["matched_windows"] = case["matched_windows"]
        result["reproduction"] = case.get("reproduction", {})
        result["matched_total"] = case.get("matched_total", len(case["matched_windows"]))
        result["matched_truncated"] = case.get("matched_truncated", False)
        result["evidence_source"] = "matched"
        result["context_supplied"] = bool(
            case["query"].get("context") or case["source"].get("context")
        )
        result = apply_extent_guard(result, case)
        return apply_review_policy(result, case)


def http_completer(url=None, model=None, timeout=None, api_key=None,
                   max_tokens=None, thinking=False):
    """An OpenAI-compatible chat completer, configured from the environment.

    QI_LLM_URL     full endpoint, e.g. http://host:8000/v1/chat/completions
    QI_LLM_MODEL   model name to request
    QI_LLM_KEY     optional bearer token
    """
    url = url or os.environ.get("QI_LLM_URL", "")
    model = model or os.environ.get("QI_LLM_MODEL", DEFAULT_MODEL)
    timeout = timeout or int(os.environ.get("QI_LLM_TIMEOUT", DEFAULT_TIMEOUT))
    api_key = api_key or os.environ.get("QI_LLM_KEY", "")
    max_tokens = max_tokens or int(os.environ.get("QI_LLM_MAX_TOKENS", 1200))

    if not url:
        raise ValueError("no classifier endpoint: pass url or set QI_LLM_URL")

    def complete(system_prompt, user_prompt):
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        # advisory only: some builds honour enable_thinking, this one does not, so
        # the token budget below is what actually keeps a thinking model in check
        if not thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(payload).encode()

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:400]
            raise RuntimeError(f"classifier endpoint returned {error.code}: {detail}")
        except urllib.error.URLError as error:
            raise RuntimeError(f"classifier endpoint unreachable at {url}: {error.reason}")

        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected completion payload: {str(payload)[:400]}")

        content = message.get("content") or ""
        if not content.strip():
            if message.get("reasoning_content"):
                raise RuntimeError(
                    "the model produced no answer: it spent the whole token budget "
                    "thinking. Raise QI_LLM_MAX_TOKENS (default 1200)."
                )
            raise RuntimeError("the model returned an empty message")

        return content

    return complete
