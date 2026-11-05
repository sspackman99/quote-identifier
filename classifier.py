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

MATCHED_WINDOW_LIMIT = 8
DEFAULT_MODEL = "local-classifier"
DEFAULT_TIMEOUT = 180  # this model can take a while to think
REVIEW_CONFIDENCE_FLOOR = 0.75
CONTRADICTION_SIMILARITY = 0.85

SYSTEM_PROMPT = """You classify the relationship between two scriptural passages.

Answer with ONE of these labels:

- quotation: the query reproduces the source substantially verbatim, at length.
- partial_quotation: the query reproduces a shorter run of the source verbatim.
- paraphrase: the query restates the source's statement or idea in different words.
- allusion: the query echoes or evokes the source without restating it closely.
- interpretation: the query explains, applies or comments on the source.
- shared_vocabulary: similar wording, but no evidence that one depends on the other.
- unrelated: no meaningful relationship.

How to decide:
1. The MATCHED evidence below is the only thing that can support quotation or
   partial_quotation. Quote from it when you claim verbatim reuse.
2. SURROUNDING CONTEXT is provided to help you understand both passages. Similarity
   that exists only in the surrounding context is never a quotation.
3. A high similarity score on its own is not evidence of dependence. Two passages can
   share idiom, covenant formula or prayer language without either drawing on the other.
4. If the query extends or explains the source, prefer interpretation over paraphrase.
5. If the evidence is genuinely ambiguous, still choose the best label AND set
   needs_review to true. Do not inflate confidence to seem decisive.

Return JSON only, with no prose around it:

{
  "relationship": "<one label above>",
  "confidence": <0.0 to 1.0>,
  "query_span": "<verses in the query passage that carry the relationship>",
  "source_span": "<verses in the source passage being used>",
  "explanation": "<one or two sentences>",
  "evidence": ["<short quoted phrases from the MATCHED evidence>", "..."],
  "needs_review": <true or false>
}

If the relationship is "unrelated", evidence may be empty and the spans may be "".
"""


class ClassifierResponseError(ValueError):
    """The model returned something that cannot be used as a classification."""


def build_case(query, source, matched_windows, retrieval=None, context_radius=2):
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

    return {
        "relationship": relationship,
        "confidence": round(confidence, 4),
        "query_span": str(payload.get("query_span", "")).strip(),
        "source_span": str(payload.get("source_span", "")).strip(),
        "explanation": explanation,
        "evidence": evidence,
        "model_needs_review": bool(payload.get("needs_review", False)),
    }


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
        result["matched_windows"] = case["matched_windows"]
        result["matched_total"] = case.get("matched_total", len(case["matched_windows"]))
        result["matched_truncated"] = case.get("matched_truncated", False)
        result["evidence_source"] = "matched"
        result["context_supplied"] = bool(
            case["query"].get("context") or case["source"].get("context")
        )
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
