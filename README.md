# quote-identifier
A repository to identify moments in the scriptures where a speaker/author quotes another

## Running

The scripture text is not in this repository. Put your copy at `data/bom.txt`, pass
`--corpus PATH`, or set `QI_CORPUS`; see `data/README.md`. With no argument the pipeline
looks for `data/bom.txt`, then `./bom.txt`, and exits with an error naming both.

On a machine with a GPU (embeddings computed locally with sentence-transformers):

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    python main.py --corpus data/bom.txt --output top_matches.csv

On a CPU-only machine, with embeddings served by a GPU host over an
OpenAI-compatible `POST /v1/embeddings` endpoint:

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python main.py --corpus data/bom.txt

With `QI_EMBEDDINGS_URL` set (or `add_embeddings(..., url=...)` passed), torch and
sentence-transformers are never imported, so `numpy pandas faiss-cpu datasketch` is the
whole local requirement. Vectors are L2-normalized client-side, so cosine similarity
stays a plain inner product.

One index must be built from ONE embedding model. Mixing vectors from different models
(e.g. `all-mpnet-base-v2` locally and `embeddinggemma-300m` from a server) inside a
single run is meaningless, so pick one per run.

## Corpora

Corpus text is external (see `data/README.md`). Each source has its own parser, and all of
them emit the same canonical record:

    corpus, book_id, book, chapter, verse, text, source_ref, source_line

- `corpus`: corpus id, e.g. `bom`, `kjv`
- `book_id`: stable identifier — `isaiah` from any source, whatever its heading style
- `chapter`, `verse`: integers
- `source_ref`, `source_line`: what the file actually printed and where, for audit

Registered parsers (`corpora.CORPUS_PARSERS`):

    bom   Project Gutenberg Book of Mormon (book titles, chapter headings, inline
          4 Nephi verse markers)
    kjv   one tab-separated verse per line: "Genesis 1:1<TAB>In the beginning..."

Parse one corpus through its own parser:

    python main.py --corpus data/bom.txt --corpus-id bom
    python main.py --corpus data/kjv.txt --corpus-id kjv

A line that is neither a header nor a verse raises `CorpusFormatError` naming the line
number, instead of being dropped — a truncated file must not look like a shorter canon.

### Cross-corpus retrieval

`cross_corpus.py` windows both corpora into one index and asks what a chapter of one
quotes from the other:

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python cross_corpus.py --bom data/bom.txt --kjv data/kjv.txt \
        --query "Mosiah 14" --top 10 --json cc_mosiah14.json

It reports, for each target chapter, how many candidate windows of the query chapter
pointed at it, with best and mean similarity, plus whether Isaiah 53 was retrieved.
Common trigrams are filtered once over the pooled corpus so both sides lose the same
boilerplate. MinHash is not used here: this stage is about embedding retrieval.

### Retrieval evaluation

`evaluate_retrieval.py` scores the retriever against a labelled set
(`evaluation_pairs.py`). Each pair carries the source and target **verse spans**
(`whole` or `21-24`), a confidence, a note when the label rests on a judgement, and a
`reviewed` flag showing whether the label has been checked against the text — 10 of 44
have been so far.

  quotation          21 pairs: whole chapters (Isaiah 48-49, 50-51, 2-14; Isaiah 53,
                     54; Malachi 3-4)
  partial_quotation   3 pairs: Mosiah 12:21-24, 3 Nephi 20:36-40, Moroni 10:31
  paraphrase          4 pairs: 3 Nephi 12-14 / Matthew 5-7, Moroni 7:45-48 / 1 Cor 13:4-8
  allusion            2 pairs: Alma 7:10-12, 3 Nephi 19:20-23
  interpretation      3 pairs: Mosiah 15, 2 Nephi 2:15-21, Alma 42:2-6 — contextual
                      dependence on a source the discourse already quotes or retells.
                      Scored and printed SEPARATELY, excluded from retrieval recall:
                      an exposition of Isaiah 53 does not "retrieve" it.
  shared_vocabulary   3 controls with KJV-shaped vocabulary and no source chapter
  unrelated           8 controls with no KJV source at all

The dev/test split is assigned **per source family**, not per query, so neighbouring
chapters of one quotation block cannot straddle it. Thresholds are swept on dev only;
test is reported at a threshold fixed in advance (0.90) with precision, recall and
false-positive rate, where an accepted control is a false positive and the count of
unreviewed labels is printed alongside.

    QI_EMBEDDINGS_URL=http://10.77.1.117:8001/v1/embeddings \
    QI_EMBEDDINGS_MODEL=embeddinggemma-300m \
    python evaluate_retrieval.py --cache-index .eval_cache --json eval_result.json

It builds one pooled index (cached in `.eval_cache/` after the first run, so re-scoring
is seconds rather than minutes). For every query chapter the same matched-window evidence
is ranked by two rules — `chapter` (window-hit count, then best similarity: the original
rule) and `span` (best similarity, then distinct non-overlapping supports, then coverage,
with candidates within `--tie-tolerance` treated as ties so one lucky window cannot beat
real support) — and both are reported, so a scoring change can be compared like-for-like.

Each candidate keeps its evidence: best similarity, number of distinct supporting verse
ranges, coverage of the candidate chapter, and the source- and query-side verse ranges
doing the pointing. Three measurements follow from that: source **chapter** recall, source
**span** recall (does the matched range touch the annotated verses), and target-span
localization (does the query-side range touch the verses that do the quoting). Annotations
are used only to score the run; retrieval is never told where the quotation is.

Candidate chapters are grouped into deterministic similarity BANDS rather than compared
pairwise: the highest remaining score anchors a band, every candidate within
`--tie-tolerance` of that anchor joins it, and the band is ordered by support, coverage,
then similarity. Pairwise tolerance is not transitive (A ties B, B ties C, A does not tie
C), which would make the ordering depend on sort behaviour; bands are a total order.

Each candidate also keeps its paired evidence: which query window matched which source
window and how strongly, not just the two independently merged ranges. That is what later
verse-level localization and classification will need.

A pair may label several independent sources. Each target carries its own reference, verse
span, relationship kind, confidence, review status and query span, and is scored
separately; recall is reported twice, crediting any labelled source and the primary target
only, so multi-source labels cannot silently inflate the headline number. Mosiah 12 quotes
Isaiah 52:7-10 *and* the Decalogue of Exodus 20:2-4 in the same chapter; crediting one
source scored a correct retrieval as a miss.

Every control was reviewed against the text and the closest approach recorded. Max control
similarity is 0.8169 (Ether 1, the Abraham/Isaac land-and-seed formula). Two chapters that
began as controls, Helaman 5 and Helaman 11, were moved to low-confidence allusions: they
pattern on the transfiguration and on Psalm 85 without reproducing either.

Credit is span-valid. A candidate chapter earns recall credit only when its matched source
range actually touches that target's annotated span, and localization credit only when the
matched query range touches the verses doing the quoting. A high rank with the wrong verses
therefore earns nothing, and a lower-ranked expected source with the right verses gets the
credit instead. The threshold metrics apply the same rule: a top-1 hit counts as a true
positive only if its reference and its matched range both satisfy a labelled target.

Recall is reported in distinct tiers rather than one number: chapter retrieved, source span,
source span plus query localization, and the primary target on its own. Paired evidence is
kept in full (no top-ten cut); the row records `source_pairs_total` so a consumer can see it
is complete.

Measured on the current set (69,929 windows, 50 neighbours, tie tolerance 0.01). The label
review has moved three chapters out of the control set over its two passes, so these
denominators differ from earlier runs and the numbers are not directly comparable to them:

    tier                      dev @1   test @1   test @5   test @20
    chapter retrieved          20/20    8/13      10/13     11/13
    source span                20/20    11/13     12/13     12/13
    span + query localization  20/20    11/13     12/13     12/13
    primary source span        20/20    10/13     12/13     12/13

    quotations + partials      16/16     7/8       8/8       8/8
    allusions                  -         4/5       4/5       4/5

    chapter ranking, for comparison: chapter retrieved 8/13 @1, 10/13 @5

Quotations and partial quotations are always published separately from allusions: they are
different kinds of textual relationship, and pooling them would hide which one moved.

Regression cases, chapter rank -> span rank: Moroni 10 30 -> 2, Mosiah 12 3 -> 1 (credited
to its second source, Exodus 20), 3 Nephi 20 1 -> 1, Mosiah 14 1 -> 1. The single-verse
quotation in Moroni 10:31 is the case the change exists for. Ether 1 -> Genesis 26 is the
clearest illustration of the same effect in the other direction: chapter rank 11, span rank
1, best 0.8169 — hit-count ranking buries it behind chapters with more scattered matches.

Held out at the 0.90 threshold fixed in advance: 9 accepted, 7 true positives (6 of them
primary), 0 control false positives, 2 accepted context pairs, precision 0.78, recall 0.54,
false-positive rate 0.00 over 8 reviewed controls. The threshold remains untuned, and eight
controls is not a reliable estimate of false-positive behaviour in general.

## Classification rubric

`RUBRIC.md` is the authority for what each of the seven labels means: version 1.0,
with the two distinctions that decide borderline cases (extent of reproduction decides
quotation versus partial quotation; the relation between statements decides paraphrase
versus interpretation) and a worked example for each boundary the classifier prompt
teaches. The version is written into every classification result and into the
evaluation JSON, because results under different rubrics are not comparable.

Relabelling is a version bump plus a re-label pass, never an edit in place: affected
pairs keep their previous label in `label_history` with the reason, and a pair may carry
a `secondary_relationship` when two relationships both apply (Mosiah 15 and Alma 42 are
interpretations with an embedded partial quotation).

## Measuring extent (deterministic, no model calls)

The rubric's extent rule needs a number, and embeddings cannot supply one: matched
windows spread across a chapter, so chapter-level coverage saturates at 1.00 even when
three verses are quoted. `coverage.py` measures reproduction from the wording instead.
For a source chapter it asks, verse by verse, what share of that verse's four-word
sequences also occur in the query, then reports:

    reproduced_share   share of the source chapter's verses that are reproduced
    longest_run        longest run of consecutive reproduced verses
    query_share        share of the query's wording that comes from the source
    extent reading     quotation / partial_quotation / no reproduction, with the reason

Measured over the labelled set (`coverage_report.py`):

    label                n    reproduced share (min / median / max)
    quotation           24    65% / 100% / 100%
    partial_quotation    3     0% /  27% /  73%
    paraphrase           1             23%
    interpretation       3     0% /   0% /   8%
    allusion             5     0% /   0% /   0%

It agrees with the quotation/partial label on 25 of 27 pairs, and the two disagreements
are both worth a human ruling rather than a code change:

    Moroni 10 -> Isaiah 52   labelled partial_quotation, measured 0% reproduced.
                             The wording is close but not verbatim ("awake and arise
                             from the dust" against "awake, awake, put on thy strength"),
                             so the rubric's reading is paraphrase.
    3 Nephi 20 -> Isaiah 52  labelled partial_quotation, measured 73% of the chapter
                             reproduced in a run of 7 verses, so the rubric's reading is
                             quotation.

Both labels are left as they are pending that ruling; the measurement is reported with
them rather than silently overriding them.

The classifier receives the measurement as its own clearly labelled block — "MEASURED
REPRODUCTION (computed from the text, not a model judgement)" — and the extent guard now
uses it in three tiers instead of the saturated chapter coverage: a `quotation` above 60%
stands; between 10% and 60% it is recorded as `partial_quotation`; below 10% the wording
is not there at all, so it becomes `paraphrase` and is flagged for review. Downgrade only,
never a promotion.

On the eight disagreement pairs the guard removed the worst error class — a quotation
claimed where almost nothing is reproduced — from two cases to one (3 Nephi 19 at 0%
reproduction now reads paraphrase with a review flag). Exact accuracy stayed at 1/8
because the remaining calls are one-grade boundary disputes where the measurement
supports either side.

## Classification stage

Retrieval answers "these two passages look alike". `classifier.py` answers "what kind
of relationship is that, and how sure are we", and it is deliberately separate from
`main.py` so a retrieval run does not pay for a model call.

The module is provider-agnostic and does no I/O of its own:

    case = build_case(query={...}, source={...}, matched_windows=[...], retrieval={...})
    result = RelationshipClassifier(complete).classify(case)

`complete` is any callable `(system_prompt, user_prompt) -> str`; `http_completer()` wraps
an OpenAI-compatible chat endpoint and reads `QI_LLM_URL`, `QI_LLM_MODEL`, `QI_LLM_KEY`,
`QI_LLM_TIMEOUT` and `QI_LLM_MAX_TOKENS`. Generation is bounded and reasoning-model
"thinking" is off unless `--thinking` is passed, because this is a short-answer task.

Two kinds of evidence are passed and never merged: the MATCHED window pairs retrieval
actually found, and SURROUNDING CONTEXT on both sides. The system prompt says that only
matched evidence can support a quotation, and that context-only similarity never can —
otherwise a broad contextual resemblance gets reported as verbatim reuse.

`classify()` returns structured fields, not prose:

    relationship   one of the seven taxonomy labels
    confidence     0..1, as reported by the model
    query_span     verses in the query carrying the relationship
    source_span    verses in the source being used
    explanation    one or two sentences
    evidence       short quotes drawn from the matched evidence
    needs_review   set by the model OR by policy, with review_reasons recorded

The policy escalates regardless of what the model claims: confidence below 0.75, a model
flag, "unrelated" at similarity 0.85 or above, and any dependent label without verse
spans. Malformed answers are errors rather than a silent `unrelated`: a label outside the
taxonomy, a confidence outside 0..1, a missing explanation, or a dependent label with no
quoted evidence all raise `ClassifierResponseError`.

`classify_evaluation.py` runs the classifier over `evaluation_pairs.py` and scores it
against the labels. `--candidate retrieved` classifies what a user would actually be
shown, so errors mix retrieval and classification; `--candidate expected` classifies the
labelled source chapter and measures classification by itself. The confusion matrix is
reported for retrieval-correct pairs, because the classifier cannot be blamed for a
candidate it was never given.

    python classify_evaluation.py --limit 1 --only "Moroni 10" --dry-run
    QI_LLM_URL=http://host:8000/v1/chat/completions QI_LLM_MODEL=my-model \
        python classify_evaluation.py --candidate expected --json classification.json

Nothing here changes retrieval: scores, threshold, spans and ranking are untouched, and
the classifier is not yet wired into `main.py`'s CSV.

### First measured run

Qwen3.5-9B-Q4-Vision on ozai, `--candidate expected`, 34 of 35 labelled positives
(the one skip is Alma 7, whose source chapter retrieval never surfaced at all):

    expected kind        n    correct   what it said instead
    quotation           21    21        -
    partial_quotation    3     1        quotation x2
    paraphrase           3     0        quotation x3
    allusion             4     1        paraphrase x3
    interpretation       3     0        quotation x2, paraphrase x1
    total               34    23

    source-span agreement 2/2 where both sides gave one
    flagged for review    1/34

The errors are concentrated on the boundaries rather than on facts. Full-chapter
quotations are recognised perfectly; the disagreements are (a) quotation versus
partial quotation, which this model treats as one category, and (b) paraphrase and
interpretation, which it tends to promote to quotation when the wording really is
close.

Two of those turned out to be label errors rather than classifier errors, and they
were the most useful thing the run produced. Under rubric 1.0 the labels moved: 3 Nephi
12, 13 and 14 are `quotation` (49%, 84% and 86% of the source's four-word sequences
reproduced), with the old `paraphrase` label preserved in `label_history`; Mosiah 15 and
Alma 42 stay `interpretation` with `partial_quotation` recorded as the secondary
relationship, because the exposition dominates and only a phrase is quoted.

This section reports the pre-1.0 run. The v1.0 run against the relabelled set is
reported below it once it completes.

The single review flag was a genuine error the policy caught: with the wrong query
verses in the prompt (see below) the model called Mosiah 12 -> Exodus 20 `unrelated`
at confidence 1.00, and `unrelated` at similarity 0.95 escalated it automatically.

### After the rubric: what changed and what did not

Relabelling under rubric 1.0 lifted the expected-kind score from 23/34 to 26/34, but
three of those gains are the relabelled Matthew passages, so the honest reading is that
the labels moved toward the model on evidence (49% / 84% / 86% of the source's four-word
sequences reproduced), not that the model improved.

Teaching the extent and dominance tests in the prompt changed nothing measurable on the
eight disagreement pairs (0/8), and the model kept describing excerpts as "a substantial,
continuous portion of the source" — it applies the quotation definition without checking
the size of the source passage. A deterministic guard now applies the rubric's extent
rule where retrieval can measure it: a `quotation` whose matched evidence covers less
than 60% of the cited source passage is recorded as `partial_quotation`, with
`guard_original_relationship` and `guard_note` preserving what the model said and why it
was overridden. The guard only ever downgrades. It fixed one of the eight (Mosiah 12 ->
Exodus 20).

What remains is concentrated: excerpts called quotations when the match covers most of a
short chapter (3 Nephi 20, 3 Nephi 19), one paraphrase/partial boundary (Moroni 7), and
dominance questions where exposition and quotation are interleaved (Mosiah 15, Alma 42,
2 Nephi 2). A chapter-level coverage proxy cannot see the difference between "cited the
whole of a short psalm" and "quoted three verses out of twenty-six"; span-level coverage
against the cited verses is the obvious next measurement.

Retrieved mode, sampled at 8 pairs, behaves differently and as expected: the two
controls shown to it were judged correctly (Mosiah 4 `unrelated` 0.95, 3 Nephi 4
`paraphrase` 0.65 with a review flag), and the positives it got wrong were ones where
retrieval had already chosen the wrong chapter (Moroni 10 was shown 1 Corinthians 12
and answered `unrelated`). Two requests out of 42 exceeded the client timeout.

A harness bug this run exposed: a multi-source pair was given the primary target's
query verses for every source, so Mosiah 12 -> Exodus 20 was presented as the
Isaiah 52 verses against the Decalogue. Each target now brings its own query span,
and the same pair classifies as `quotation` 0.98.

## Output

`top_matches.csv` holds the highest-scoring candidate pairs:

    score, methods, text_1, book_1, chapter_1, start_verse_1, end_verse_1,
    text_2, book_2, chapter_2, start_verse_2, end_verse_2

`methods` lists every retriever that found the pair (`embedding`, `minhash`). Each pair
appears once: the two retrievers and both directions (A->B, B->A) are collapsed into one
row. `--no-dedupe` restores the raw per-method, per-direction rows.

Candidate pairs are still window pairs, and windows overlap, so the same passage pair can
legitimately appear more than once with different verse spans. Collapsing those belongs
with context expansion, where a span means something.

## Windowing

30-token windows with 15-token overlap, built per chapter (`chunking.py`):

- a window never crosses a chapter boundary;
- a chapter shorter than the window yields one window holding the whole chapter;
- the last window is anchored to the chapter end when the step grid would leave a tail,
  so no trailing text is left uncovered.

`test_window_coverage.py` checks those as geometric properties over many chapter lengths
and window/overlap settings, independently of the similarity pipeline.

## Filtering common phrases

3-grams that are too common are stripped before matching (`filtering.py`). Two measures
exist, and their thresholds are NOT interchangeable because the denominators differ:

- `document frequency` (default): how many chapters contain the trigram, counted once per
  chapter however often it repeats. A document is `(corpus, book, chapter)`, so the same
  book/chapter in a different corpus is a different document.
- `occurrence count` (legacy, `--filter-metric occurrence`): every position in every
  window. Overlapping windows and repeats inside a passage inflate it. Retained as a
  separate statistic so the two can be compared on the same corpus.

Measured on the Book of Mormon (239 chapters, 17,857 windows, 535,710 window tokens):

    metric                                    phrases   tokens removed
    document >20% of chapters (default)            76   52,059  (9.7%)
    occurrence >1% of windows (legacy)             54   45,927  (8.6%)
    document >1% of chapters (same number)     12,341  281,655 (52.6%)

That last row is why `MIN_DOCUMENT_FREQUENCY` is 0.20 and not 0.01: on this corpus 1% of
239 chapters means "in 3 chapters", which is a far looser bar than 1% of 17,857 windows.

Effect on the top 750: 606 of 665 reference pairs are shared; 59 appear only under the
document metric and 58 only under the legacy one, including some high-scoring pairs in
both directions. `--filter-metric occurrence` reproduces the old behaviour exactly.

## Tests

    python -m pytest -q

