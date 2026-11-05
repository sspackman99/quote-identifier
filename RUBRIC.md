# Classification rubric — version 1.0

The classifier, the evaluation labels and any future re-labelling all use this
vocabulary. The version string is recorded in every classification result and in the
evaluation JSON, so a change in the rubric is visible rather than silent.

## Labels

| Label | Definition | Key test |
| --- | --- | --- |
| `quotation` | A passage reproduces a substantial, continuous portion of the source, with wording substantially preserved. | Is it essentially a direct quotation of the source passage? |
| `partial_quotation` | A shorter, continuous, substantially verbatim excerpt from a larger source passage. | Is the wording directly reproduced, but only a portion of the source? |
| `paraphrase` | The source's substantive proposition or meaning is restated in different wording, without merely explaining its implications. | Is the same assertion being made in different words? |
| `allusion` | An identifiable reference to a source through distinctive language, imagery, structure, or a recognizable combination of ideas, without reproducing or restating the passage sufficiently to qualify as a quotation or paraphrase. | Is the source being evoked rather than directly reproduced or restated? |
| `interpretation` | The passage explains, applies, expands, draws an inference from, or develops the meaning of another passage. | Is it explaining what the source means or implies, rather than restating what it says? |
| `shared_vocabulary` | Similar words or conventional expressions occur without persuasive evidence of a specific textual relationship. | Could this similarity reasonably occur independently? |
| `unrelated` | No meaningful textual relationship is supported. | Is there no identifiable connection? |

## Two distinctions that decide most borderline cases

**1. Quotation versus partial quotation is about the extent of the reproduced text,
not its importance.** A short but exact quotation is `partial_quotation` even when it
is the most recognizable sentence in the source. A passage that reproduces the source
substantially is `quotation`. There is no verse-count threshold: one verse can carry a
substantial quotation and several verses can carry only a short excerpt. Judge the
continuous wording actually reproduced.

**2. Paraphrase versus interpretation is about the relation between the statements.**
`paraphrase` says substantially the same thing in different words. `interpretation`
explains, applies or draws conclusions from what the source says. A passage can do
both; then the dominant relationship decides the label and the secondary relationship
is preserved in the explanation and in the label notes, rather than being forced into a
single flat answer.

## Boundary examples used in the classifier prompt

| Boundary | Leans one way | Leans the other |
| --- | --- | --- |
| short excerpt vs substantial reproduction | Moroni 10:31 -> Isaiah 52:1-2: two verses reproduced exactly inside a long chapter = `partial_quotation` | Mosiah 14 -> Isaiah 53: the whole servant song, in order = `quotation` |
| restating vs explaining | Moroni 7:45-48 restates 1 Corinthians 13's description of charity in adapted wording = `paraphrase` | Mosiah 15 explains who the servant of Isaiah 53 is, quoting a phrase or two = `interpretation` |
| allusion vs shared vocabulary | Helaman 11:11-12 evokes Psalm 85's plea to turn away anger = `allusion` | Alma 43:46 restates a legal principle any ancient law code could carry = `shared_vocabulary` |
| interpretation containing a quotation | Alma 42 explains the Fall and names the cherubim and flaming sword of Genesis 3:24 = `interpretation` (the quotation is secondary) | a block of source verses introduced and then commented on, where the block dominates = `quotation`, with the comment secondary |
| several sources at once | Mosiah 12 quotes Isaiah 52 in 12:21-24 and the Decalogue in 12:34-36; each pair is judged against its own source and its own query verses | — |

## Consequences already applied (v1.0)

Measured evidence: share of the source's four-word sequences that also occur in the
query chapter. It is a chapter-level indicator, so it is evidence for a label, never a
label by itself — a short exact quotation inside a long chapter scores near zero.

| Pair | v1.0 label | Was | Evidence |
| --- | --- | --- | --- |
| 3 Nephi 12 -> Matthew 5 | `quotation` (with additions) | `paraphrase` | 49% of Matthew 5's 4-grams reproduced, query longer than source |
| 3 Nephi 13 -> Matthew 6 | `quotation` | `paraphrase` | 84% reproduced |
| 3 Nephi 14 -> Matthew 7 | `quotation` | `paraphrase` | 86% reproduced |
| Mosiah 15 -> Isaiah 53 | `interpretation` (secondary: `partial_quotation`) | `interpretation` | 3% reproduced: a phrase or two inside exposition |
| Alma 42 -> Genesis 3 | `interpretation` (secondary: `partial_quotation`) | `interpretation` | 7% reproduced: the cherubim and flaming sword |

## Changing the rubric

A rubric change is a version bump plus a re-label pass, not an edit in place. Keep the
previous label in `label_history` on each affected pair: the original judgement, its
rubric version, and why it moved. Results produced under different versions are not
directly comparable, and reporting should say which version a number belongs to.
