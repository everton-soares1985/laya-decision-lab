"""Validate the private knowledge seed. Run with python -X utf8 scripts/validate_knowledge.py."""
import json
import logging
from pathlib import Path

# ==================== CONFIGURATION ====================
ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "data" / "knowledge" / "profile_v1.json"  # Private, ignored by Git.
STATUSES = {"confirmed", "needs_clarification"}  # No automatic promotion.
# =======================================================


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    doc = json.loads(PROFILE.read_text(encoding="utf-8"))
    if doc["schema_version"] != "1.0" or not doc["pending"] or not doc["policies"]:
        raise ValueError("Invalid knowledge schema")
    seen = set()
    for fact in doc["facts"]:
        if fact["id"] in seen or fact["status"] not in STATUSES:
            raise ValueError("Duplicate ID or invalid status")
        seen.add(fact["id"])
        if not fact["source"]["evidence"] or not fact["constraints"] or fact["value"] is None:
            raise ValueError("Missing evidence, constraints or value")
        if fact["source"]["kind"] == "local_artifact":
            artifact = (ROOT / fact["source"]["evidence"]).resolve()
            if not artifact.is_relative_to(ROOT) or not artifact.is_file():
                raise ValueError("Evidence artifact missing or outside project")
    ready = sum(f["status"] == "confirmed" for f in doc["facts"])
    logging.info("knowledge_valid records=%d confirmed=%d clarification=%d pending=%d",
                 len(seen), ready, len(seen)-ready, len(doc["pending"]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("knowledge_validation_failed")
        raise
