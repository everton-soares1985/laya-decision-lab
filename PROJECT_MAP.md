# Source map

Independent experimental question-classification, retrieval and answer engine.
The LinkedIn browser consumer is maintained separately. See docs/ROADMAP.md for
limitations and docs/ENGINEERING_JOURNAL.md for the development history.

## Reuse these modules before creating new scripts

| Module | Responsibility | Prerequisites |
|---|---|---|
| scripts/prepare_dataset.py | Deduplication, review, topic/context preparation | Authorized captures; LAYA_CAPTURE_SOURCE |
| scripts/run_probe.py | Bounded CPU training supervisor | Windows venv, upstream Laya, dependencies |
| scripts/train_probe.py | Head-only or full-parameter training | Model, reviewed dataset, explicit training invocation |
| scripts/evaluate_checkpoint.py | Saved-checkpoint diagnostics | Checkpoint and matching dataset/schema |
| scripts/retrieve_knowledge.py | Deterministic evidence and scope checks | Profile; fictional self-test |
| scripts/retrieve_openrouter.py | Catalog routing, paced transport, API validation | Explicit API opt-in; optional local catalog/settings |
| scripts/retrieve_hybrid.py | API embeddings, local Qdrant, BM25, rank fusion | Profile, embedding cache or authorized API access |
| scripts/generate_answer.py | Closed-answer rules and grounded open drafts | Profile; fictional self-test |
| scripts/run_pipeline.py | Batch execution, budgets, resume, public demo | Public demo: standard library only |
| scripts/linkedin_adapter.py | JSON-lines worker and field/job boundary | Full runtime for serve; fictional offline self-test |
| scripts/validate_knowledge.py | Private fact schema checks | User-authored profile |

## Public files

- examples/public/profile.json and questions.json: nine-case fictional rules demo.
- examples/public/regression_profile.json: invented legacy-regression profile.
- tests/test_public_demo.py: offline tests, private-read/network guards.
- .github/workflows/public-demo.yml: Windows/Linux standard-library CI.
- docs/: architecture, training, retrieval, evidence, reproduction, media and release boundaries.
- screenshots/: authentic stills; conceptual artwork belongs in docs/assets/.

## Local-only artifacts

data/, runs/, cache/, vendor/, .venv/, .env, Laya_Midia/, PROJECT_LOG.md and old
examples/*.json are not distributed. Historical operational source-map notes were
preserved locally before this public map replaced them. No private profile or
checkpoint is required by the public rules demo.

Private compatibility settings in data/runtime.local.json retain existing local
capture/environment paths. data/knowledge/catalog.local.json retains the owner's
exact catalog descriptions, preserving retrieval/cache inputs. Neither file is
committed. Public defaults contain schema descriptions, not applicant answers.
