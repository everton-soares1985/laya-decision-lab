# Engineering journal: from a fast probe to a hybrid system

This is a curated public account. Original private logs are retained locally but
excluded from publication. It separates observed outcomes from planned work and
does not expose candidate answers, application IDs or browser credentials.

## The initial problem

The consumer automation encountered multilingual questions with variable wording,
conditional fields, long dropdown labels and missing context. Some questions were
answerable from known facts but the system still stopped. The project started by
asking whether a small specialized classifier could make interpretation more
consistent, while also providing a concrete machine-learning portfolio artifact.

## 20 October: establish that training is real

Short synthetic probes verified loading, optimization and checkpoint persistence
on the available CPU. These quick runs were intentionally small; their short
duration did not establish that a useful domain model had been trained.

The next stages imported real captured questions, distinguished capture counts
from unique examples, and expanded annotation. Head-only and full-parameter runs
were kept separate. A synthetic augmentation experiment did not improve the real
evaluation score. That negative result is part of the project, not discarded.

A subsequent audit reviewed 154 labels, excluded 17 with individual reasons and
retained 137 real examples. The selected question-only topic experiment trained
on 113 examples for four epochs. A separate evaluator reloaded checkpoints to
verify that saved artifacts, rather than in-memory state alone, reproduced results.

## 20–22 October: improve the question, not just the score

Error analysis showed that the taxonomy conflated topic, sensitivity, consent and
follow-up context. Primary topics were separated from review flags. The current
model learns the topics; it does not automatically learn those flags.

Context-enriched examples and validation ablations were explored. The recorded
question-only validation comparison was 9/12 versus 7/12 with enriched context in
one ablation, so context was not adopted as a universal improvement. The small
inspected sample prevents strong claims either way.

An independent annotation review was requested through Qwen Chat, manually
exchanging files rather than calling a Qwen API. Reviewer suggestions highlighted
ambiguous categories and conditional fields. This was a review aid, not automatic
ground truth or a model-performance measurement.

Further training was paused. The important remaining problem was not simply
classification: the system needed facts, scope and an answer contract.

## 22 October: start with an evidence baseline

A private knowledge profile and deterministic retrieval baseline were created.
Explicit records could answer questions such as availability or salary while
preserving units and provenance. The first retrieval-only result intentionally
returned evidence, not permission to submit an application.

API routing experiments encountered provider/transport failures. Pacing, bounded
retries, resume behavior and validation diagnostics were added. Retry accounting
was kept separate from successful outputs so a failed request would not disappear
from operational observation.

## 23 October: connect the pipeline and hybrid retrieval

A batch entrypoint combined the frozen classifier, known facts and optional LLM
generation. An eight-item integrated local demonstration recorded six direct
answers, one missing-information result and one draft, with one reported API
attempt. This demonstration is not a general answer-rate benchmark.

The retrieval experiment added OpenRouter embeddings, Qdrant local cosine search,
SQLite FTS5 BM25 and RRF. Abstract descriptions, not the full candidate profile,
formed the searchable catalog. The original eight indexed descriptions expanded
as reviewed facts became available.

## 24 October: controlled integration and real manual QA

The adapter first ran in preview: read fields, compare resolutions, make no browser
changes. It later exposed a persistent worker protocol to the separate LinkedIn
automation, keeping environments isolated and using the consumer's existing
submission gates and ledger.

Private project facts were clarified to describe AI-assisted implementation and
the owner's manual QA, log analysis, retesting and selector work. Unsupported
customer deployment claims and invented metrics were not added.

Manual QA exposed concrete gaps:

| Observation | Root cause / lesson |
|---|---|
| `Basic` did not fill `Basic. Example: ...` | An answer concept must be mapped to the actual option label |
| English level known but a boolean question unanswered | Classification and fact rendering are separate capabilities |
| One year assigned to a broad skill | User-authorized default was used, not verified experience retrieval |
| Country OR question unresolved | Logical membership needed explicit handling |
| Consent select unresolved | Finding a control is different from having authorized content |
| Submission clicked but not confirmed | Operational uncertainty must not be relabeled as success |

Some fixes were made in the external consumer, including confirmed education
answers, option matching and conversational-language handling. The Lab alone does
not inherit those code changes. This boundary must stay explicit in portfolio
claims and future integration tests.

## 25 October: publication preparation

The initial Publisher audit returned 10% readiness. That was a publication hygiene
score: missing Git initialization, README, license/security material and release
structure. It was **not a rating of model quality**.

Three existing Lab self-tests passed without network calls: pipeline budgeting,
hybrid retrieval and adapter contracts. The public documentation now records
architecture, training evidence, privacy boundaries, remaining packaging gaps
and visual explanations. Original videos remain private pending full review.

## Lessons worth showing in a portfolio

Publication follow-up: the fictional rules-only demo passed nine authored cases
and four standard-library tests in a fresh Windows environment. A source review
also found profile-derived expectations in legacy self-tests; passing a credential
scan did not establish privacy readiness. The release scope now tracks these
findings separately. Selected training metrics have a machine-readable, path-free
extract linked by SHA-256 to the local result artifact.

1. A successful model fine-tune does not automatically create a successful product.
2. Data quality and annotation semantics matter more than repeatedly rerunning epochs.
3. Retrieval needs an applicability layer; similarity alone can select wrong facts.
4. Form options and units are part of the answer contract.
5. Observability must distinguish missing facts, rule gaps, API failures and UI errors.
6. A reproducible public demo requires more work than a functioning private environment.
7. Clear limitations and negative results strengthen the engineering account.
