# Publication scope and review queue

This is the source-publication inventory approved on 26 October 2026.
Do not use a blanket `git add .` before reviewing source and the staged diff.

## Candidate public material

| Material | Scope | Review state |
|---|---|---|
| README files and governance | README, PT digest, LICENSE, notices, contribution/security/agent instructions | MIT first-party source; account everton-soares1985 |
| Technical guides | `docs/*.md` | Prepared; final link and claim review required |
| Visual explanations | Three `docs/assets/*.svg` | Conceptual illustrations, not screenshots of private data |
| Real training still | `screenshots/training-synthetic-expansion.png` | Visually reviewed; caption identifies the earlier expansion experiment |
| Training evidence | `docs/TRAINING_EVIDENCE.json` | Selected metrics only, source hash, no question text or personal answers |
| Fictional demo | `examples/public/*.json`, `tests/test_public_demo.py` | Offline tests passed; fixtures explicitly fictional |
| CI | `.github/workflows/public-demo.yml` | Initial source run passed on Windows and Ubuntu |
| Implementation | `scripts/*.py`, `requirements-retrieval.txt` | Source reviewed; full-pipeline artifacts remain local |

## Source-level review resolutions

Ignoring a private profile is insufficient if its values also appear in tests.
The following findings were confirmed by reading the source, independently of
the Publisher's credential-pattern scan:

- Resolved 26 October: answer-generation and adapter self-tests now use an
  independent fictional regression profile, with invented locations, compensation
  and experience values. Their original assertion coverage is preserved.
  The retrieval self-test also no longer requires private facts. A regression
  test blocks private-profile reads, network calls, credentials and model loading.
- Detailed local catalog descriptions moved to ignored `data/knowledge/catalog.local.json`.
  Public course/location descriptions are generic; existing local routing and cache
  text are preserved exactly for 34 confirmed records.
- Capture and credential paths use environment overrides or ignored local settings;
  public defaults are project-relative. Local compatibility settings preserve the
  existing integration without committing paths or keys.
- `PROJECT_MAP.md` now contains a public module map. Its historical original is
  preserved in ignored local run storage.

These are privacy and portability findings, not evidence that an API key leaked.
The public demo and self-tests run without private facts. Full model execution
still needs separately supplied data, dependencies and checkpoints.

## Keep local

Private profiles and datasets, historical QA examples, `.env` files, model weights,
embedding caches, upstream checkout, run outputs, raw recordings and historical
operational logs stay outside the initial source release. Original files must not
be deleted merely to raise an audit score. Review derived media separately.

## Final publication gate

The publication gate uses an explicit file selection, offline tests from that
selection, Publisher scanning, account verification and staged-diff review.
Source publication does not include weights, raw recordings or private datasets.
