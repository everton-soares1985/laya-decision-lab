# Roadmap and release boundaries

## Implemented locally

- [x] CPU feasibility and full-parameter training runs.
- [x] Real-question audit, topic revision and saved-checkpoint evaluation.
- [x] Confirmed-fact profile and deterministic retrieval baseline.
- [x] Hybrid Qdrant/BM25 retrieval with API embeddings and caching.
- [x] Constrained generative drafts through OpenRouter.
- [x] Batch pipeline, budgeting and persistent adapter.
- [x] External consumer integration and manual QA feedback loop.
- [x] Public architecture, training, retrieval and history documentation.
- [x] Editable diagrams, banner and Portuguese digest.

## Before calling a public release reproducible

- [x] Independent example profile with fictitious values (offline rules demo).
- [ ] Reviewed public question fixtures and artifact manifest.
- [ ] Configurable paths instead of private machine defaults.
- [x] Clean Windows environment and offline rules demo validated without dependencies.
- [ ] Full pipeline validated outside the private development environment.
- [x] Standard-library test runner and Windows/Linux CI workflow for public fixtures.
- [x] Hosted public-demo CI passed on Windows and Ubuntu.
- [ ] Full-pipeline dependency lock.
- [ ] Full review of publishable files, notices and model artifact license.
- [ ] Reviewed real screenshots and video excerpts.
- [x] Independent public source repository created and first push completed on 26 October 2026.

## Subsequent engineering

- [ ] Shared fact/option contract to reduce consumer/Lab drift.
- [ ] Distinct reasons for missing evidence, mapping gaps and API errors.
- [ ] Cost per service and per question, plus cumulative totals.
- [ ] Fresh held-out evaluation and sparse-topic coverage.
- [ ] Compare retrieval and end-to-end answers with and without the classifier.
- [ ] Evaluate whether further fine-tuning improves the actual failure modes.

## Outside this repository

Search alerts, country eligibility filtering, browser selectors, pagination,
application submission and delivery confirmation belong to the external consumer.
They can be integration case studies, but are not claimed as features implemented
inside this Lab. No additional training is scheduled by this document.
