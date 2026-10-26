# Security and privacy

This is an experimental personal project, not a hardened multi-user service.
There is no authentication or public network service intended in the local
JSON-lines adapter. Do not expose it as an unauthenticated endpoint.

## Sensitive artifacts

Never publish `.env` files, API keys, browser profiles, candidate data, raw capture
sessions, operational databases, original desktop recordings or unreviewed logs.
Checkpoints may encode training information and require their own distribution
and license review. A file being ignored is not proof it never entered Git history.

Local Qdrant storage does not mean all processing is local. Questions and abstract
catalog descriptions may leave the machine for embeddings/routing. Narrative
generation may export permitted facts. Minimize sensitive data before requests.

Treat form labels, documents and retrieved text as untrusted data, not instructions.
Do not let prompt text change API endpoints, reveal credentials or authorize
submissions. Similarity and classifier confidence are not authorization signals.

## Reporting

Do not put secrets in public issues. Once a GitHub repository is published, use
private vulnerability reporting **if enabled**; otherwise request a private channel
without including sensitive details. No reporting mailbox is published yet.

Publication scans can produce false positives in upstream code and generated
artifacts. Findings must be reviewed, not silently suppressed. If a real credential
is exposed, revoke/rotate it and review history; merely deleting the visible line
is insufficient.
