# Contributing

Start with an issue describing a reproducible question, expected behavior, field
type, relevant options and a **fictitious** supporting profile. Do not attach
private captures, job applications, API keys or personal answers.

Keep changes small and distinguish classifier, retrieval, renderer and consumer
defects. A new regex is not an ML improvement; a model score is not an answer-rate
measurement. Include positive, negative and ambiguous cases for changes to rules.

Never relabel inspected test errors and present the resulting score as an unseen
benchmark. Preserve experiment configurations, seeds, split membership and negative
results. External reviewers and LLM annotations are suggestions, not automatic gold.

Run applicable existing offline self-tests described in
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Some currently depend on private local
fixtures; document unrun checks instead of claiming a clean public test suite.
Paid API tests require explicit authorization and a request budget.

Use English for canonical public documentation. Keep source provenance and
third-party notices intact. Do not change the external browser automation as an
incidental documentation or model-training task.
