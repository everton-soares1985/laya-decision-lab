# Publication checklist

## First public source push — 2026-10-26

Repository: [everton-soares1985/laya-decision-lab](https://github.com/everton-soares1985/laya-decision-lab).
Public visibility and default branch `main` verified through the GitHub API.
Initial source commit: `182af3e`. The reviewed training still and banner were
verified present remotely. No weights, original videos or private profiles uploaded.
Hosted [run 40268741931](https://github.com/everton-soares1985/laya-decision-lab/actions/runs/40268741931)
passed on both `windows-latest` and `ubuntu-latest` for that source commit.
Older audit snapshots below describe their original point in time.

## Validation update — 2026-10-26

- Answer, retrieval and adapter self-tests now use a separate fictional profile.
- Five automated tests passed in a fresh Windows venv without installed packages,
  private data, checkpoints or cache. The nine-case public demo also passed.
- AST comparison confirmed 21 runtime function/class definitions unchanged across
  the three edited modules; legacy self-test assertion counts were preserved.
- Source review found no remaining old personal city/compensation literals in
  the answer and adapter modules. Catalog/path/history findings were subsequently
  resolved before the source push, as recorded in PUBLICATION_SCOPE.md.

## Validation snapshot — 2026-10-25

Follow-up: public baseline demo verified in a new dependency-free Windows venv:
four unit tests and nine fictional cases passed. No network, model, checkpoint,
private profile or embedding cache. CI workflow authored but not yet hosted.
A targeted Publisher-pattern scan of 30 first-party source/document/test/public
fixture files found zero matches. This does not certify all history, recordings
or upstream files as secret-free, and does not supersede a full staged-diff review.

- Ten Markdown guides checked: local links resolve and code fences are paired.
- Three SVG assets parsed and visually inspected through isolated browser renders.
- Pipeline, hybrid-retrieval and adapter self-tests passed without API calls.
- Publisher check: 57%, two errors and four warnings. This is publication hygiene,
  not model quality. Remaining errors: no Git worktree and potential-secret findings
  in the scanned tree. Findings include ignored upstream/generated files and need
  review against the actual eventual publication set; no leak is confirmed by the scan.
- Raw videos and model caches remain local; no files were deleted to improve a score.
- Mermaid blocks are authored but GitHub-side rendering has not yet been verified.

## Prepared in this documentation pass

- [x] Project name and visual identity: Laya Decision Lab.
- [x] English README and Portuguese digest.
- [x] Architecture, mind map, retrieval and training diagrams.
- [x] Historical experiment narrative with caveats and negative results.
- [x] Security/contribution guidance and first-party MIT license.
- [x] Third-party source/model distinction and license-review boundaries.
- [x] Private data, raw videos and checkpoints excluded from public Git.

## Verify before the first push

See [publication scope and source-review findings](PUBLICATION_SCOPE.md).
The credential-pattern scan does not detect every personal fact. The legacy
self-test, catalog, path and historical-map findings have been resolved. The
explicit 43-file publication candidate passed Publisher with zero errors (92%;
one warning for the uncommitted initial worktree), five tests and nine demo cases.

- [x] Authenticated account: everton-soares1985.
- [x] Public repository target: laya-decision-lab.
- [x] Publish prepared first-party MIT source under the owner's explicit push instruction.
- [x] Initialized this project on main, separately from the LinkedIn repository.
- [x] Reviewed selected files and staged diff; private artifacts excluded.
- [x] No secret findings in the selected source set; Publisher candidate check passes.
- [ ] Review upstream model card/license before distributing any weights.
- [x] Prepare fictional fixtures and verify the offline rules demo in a clean Windows environment.
- [ ] Verify the full model/retrieval pipeline from a clean clone with reviewed public artifacts.
- [ ] Inspect Mermaid rendering on the intended GitHub preview.
- [ ] Review videos fully and approve selected clips before upload.
- [x] Review one authentic still and match it to its earlier experiment's worker log.
- [x] Verify repository, README source and included media presence on the remote.

The first source push is authorized. Commit identity uses the verified account's
GitHub noreply address, not the owner's personal email. Full weights and raw
video uploads remain outside this source publication. Hosted CI and remote-link
results are recorded after push, not inferred from local tests.
