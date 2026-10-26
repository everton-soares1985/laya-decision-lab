# Reproducibility and operating guide

## Public offline demo: verified without private artifacts

Update, 26 October: the suite now includes five tests. Legacy answer, adapter and
retrieval checks use `examples/public/regression_profile.json`, a separate invented
profile with policy branches enabled for testing. All five tests passed in another
fresh dependency-free Windows venv, with only the required source modules and
fictional fixtures copied. They can also be run directly:

```powershell
python -X utf8 scripts/generate_answer.py --self-test
python -X utf8 scripts/linkedin_adapter.py --self-test
python -X utf8 scripts/retrieve_knowledge.py --self-test
```

These commands do not read the operational profile. Normal runtime commands still
use private data. The fixture's export flag tests payload validation only; the
automated tests block the network and do not generate paid drafts.

The public baseline needs only Python 3.12 and the standard library. It reuses
the existing renderer with four fictional facts and nine authored questions.
No pip install, API key, profile import, model download or checkpoint is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -X utf8 scripts/run_pipeline.py --public-demo
python -X utf8 -m unittest discover -s tests -v
```

Use `source .venv/bin/activate` on Linux/macOS. Activate an existing environment
instead of recreating it. The command writes a new `runs/public_demo_*/report.json`
on each invocation. It exits nonzero if expected statuses or answers disagree.

Expected: four supported answers and five missing-information results; all nine
checks pass, API calls zero, Laya off, hybrid retrieval off. The report never
authorizes submission. These are software regression checks, not an ML benchmark.

`--public-demo` rejects `--use-api`, `--hybrid`, `--resume`, custom input and
`--self-test`. It never falls back to the owner's profile. The test suite also
checks unknown facts and ambiguous equivalent options.

Validation on 2026-10-25 used a new virtual environment with `--without-pip`,
only four required source modules, the two public fixtures and the test module.
All four tests and nine demo cases passed. Linux CI is configured but not yet
observed on a hosted runner.

## Full-pipeline status: local reproduction, public packaging pending

Credential lookup: process `OPENROUTER_API_KEY` first; otherwise read `LAYA_ENV_FILE`,
then ignored `data/runtime.local.json` key `env_file`, then project `.env`.
Copy `.env.example` locally if API access is desired; never commit the populated file.
Capture input uses `LAYA_CAPTURE_SOURCE`, then local settings `capture_source`,
then `data/captures`. Optional `data/knowledge/catalog.local.json` overrides
description strings for existing catalog IDs. It is not an applicant-answer source.
These private compatibility files are deliberately absent from public clones.

The current system has been run in a Windows Python 3.12 virtual environment.
Do not assume the repository is yet a portable package. The following artifacts
are deliberately absent from public Git:

- Private capture datasets, labels linked to capture provenance and candidate facts.
- Trained weights and pretrained model cache.
- API credentials and operational environment files.
- Browser profiles and the separate LinkedIn application system.

This is an honest reproduction boundary, not a request to publish those files.
A larger full-pipeline fixture and configurable model/artifact paths are the next
packaging steps. The small public demo above does not replace the trained model.

## Dependency inventory

Recorded training bootstrap pins include CPU PyTorch 2.8.0, Transformers 4.57.6,
Hugging Face Hub 0.36.0, NumPy 2.2.6 and psutil 7.1.0. The local upstream Laya
checkout is installed separately. Retrieval uses `qdrant-client==1.15.1`, declared
in `requirements-retrieval.txt`. SQLite FTS5 availability is also required.

Each historical run retains its `requirements-resolved.txt` locally. These
environments have not yet been consolidated into a verified public lockfile.
Do not copy an arbitrary local freeze containing machine paths into a release.

## Known portability blockers

| Location | Current dependency | Packaging work still needed |
|---|---|---|
| `prepare_dataset.py` | Absolute path to original capture sessions | Configurable input and fictitious fixture |
| `retrieve_openrouter.py` | Environment-file path in the LinkedIn project | Independent credential configuration |
| `run_pipeline.py` | Selected historical run/checkpoint and training schema | Explicit artifact arguments or manifest |
| `generate_answer.py` | Private profile under `data/knowledge` | Public example profile and validation workflow |
| `run_probe.py` | Windows virtual-environment executable and upstream vendor checkout | Documented provisioning and platform contract |

No API key is embedded in this documentation. A referenced environment file is
still an external dependency even if it stays private.

## Existing-environment offline checks

Run from the Lab root, after activating its environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -X utf8 scripts/run_pipeline.py --self-test
python -X utf8 scripts/retrieve_hybrid.py --self-test
python -X utf8 scripts/linkedin_adapter.py --self-test
```

These were verified on 2026-10-25 with zero API requests. The hybrid check reads
the local profile and uses temporary/in-memory test structures. Passing a
self-test is not equivalent to passing an unseen ML evaluation.

## Existing-environment pipeline

With the private profile, selected checkpoint and schema already present:

```powershell
python -X utf8 scripts/run_pipeline.py --input examples/questions.json
```

Only when you intentionally authorize paid provider requests:

```powershell
python -X utf8 scripts/run_pipeline.py --input examples/questions.json --hybrid --use-api --max-attempts 3
```

The budget covers HTTP attempts, including embedding calls and retries. Offline
mode does not imply all questions can be answered: missing embeddings or facts
remain unresolved. Inspect the saved report, not only the terminal summary.

## Training reproduction: explicit action, not a quickstart default

Historical selected run command, requiring the reviewed private dataset and
installed environment:

```powershell
python -X utf8 scripts/run_probe.py --full --skip-install --watch --dataset data/real_topics_v2
```

This starts a new training run and consumes CPU/RAM. Do not run it just to view
the documentation. Recorded results are not guaranteed bit-for-bit reproducible
on other library versions or hardware. Never evaluate on newly relabeled data
and present it as the original untouched test.

## Public release acceptance criteria

- A newcomer can run a fictitious offline example without the owner's files.
- Dependency and upstream provenance are explicit.
- No automatic provider call occurs in the default demo.
- Checkpoint availability/licensing is explained rather than silently assumed.
- Local scripts do not require paths inside the private consumer project.
- Documented commands are tested in a clean environment before being called portable.

Until these criteria pass for the full pipeline, the repository is a documented
experimental codebase with a reproducible offline rules demo, not a one-command
production installation.
