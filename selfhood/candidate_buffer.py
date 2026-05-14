# candidate_buffer.py
# The pending integration queue.
#
# After each session's reflective pass, candidate patterns land here.
# A candidate graduates to LoRA integration only after appearing in
# RECURRENCE_THRESHOLD separate sessions — the consolidation gate.
#
# Structure of a candidate record:
# {
#   "id":          "sha1 of pattern text",
#   "pattern":     "Human-readable description of the latent capacity pattern",
#   "example_turns": [ { "user": ..., "assistant": ... }, ... ],
#   "sessions":    ["session_id_1", "session_id_3"],   # deduplicated
#   "occurrences": 2,
#   "status":      "pending" | "ready" | "integrated" | "rejected",
#   "created_at":  ISO timestamp,
#   "updated_at":  ISO timestamp,
# }

import json
import hashlib
from datetime import datetime, timezone
from config import CANDIDATE_STORE, RECURRENCE_THRESHOLD


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pattern_id(pattern_text: str) -> str:
    return hashlib.sha1(pattern_text.encode()).hexdigest()[:12]


class CandidateBuffer:
    """
    Thin persistence layer around a JSON file.
    For a real system this would be SQLite; JSON is fine for test scale.
    """

    def __init__(self):
        if CANDIDATE_STORE.exists():
            with open(CANDIDATE_STORE) as fh:
                self._data: dict[str, dict] = json.load(fh)
        else:
            self._data = {}

    def _save(self):
        with open(CANDIDATE_STORE, "w") as fh:
            json.dump(self._data, fh, indent=2)

    # ── Write ──────────────────────────────────────────────────────────────────

    def record_candidate(
        self,
        session_id:   str,
        pattern:      str,
        example_turns: list[dict],
    ) -> dict:
        """
        Add or update a candidate.
        If the same pattern (by hash) has been seen before, increment
        occurrence count and check against the recurrence gate.
        """
        cid = _pattern_id(pattern)

        if cid not in self._data:
            self._data[cid] = {
                "id":            cid,
                "pattern":       pattern,
                "example_turns": example_turns,
                "sessions":      [],
                "occurrences":   0,
                "status":        "pending",
                "created_at":    _now(),
                "updated_at":    _now(),
            }

        record = self._data[cid]

        # Deduplicate sessions — one session counts once per pattern
        if session_id not in record["sessions"]:
            record["sessions"].append(session_id)
            record["occurrences"] += 1
            record["updated_at"] = _now()

            # Promote to "ready" once threshold is met
            if (record["occurrences"] >= RECURRENCE_THRESHOLD
                    and record["status"] == "pending"):
                record["status"] = "ready"
                print(f"[CandidateBuffer] Pattern '{cid}' reached recurrence "
                      f"threshold — promoted to READY.")

        self._save()
        return record

    def mark_integrated(self, candidate_id: str):
        if candidate_id in self._data:
            self._data[candidate_id]["status"] = "integrated"
            self._data[candidate_id]["updated_at"] = _now()
            self._save()

    def mark_rejected(self, candidate_id: str):
        if candidate_id in self._data:
            self._data[candidate_id]["status"] = "rejected"
            self._data[candidate_id]["updated_at"] = _now()
            self._save()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_ready(self) -> list[dict]:
        """Return all candidates that have cleared the recurrence gate."""
        return [r for r in self._data.values() if r["status"] == "ready"]

    def get_pending(self) -> list[dict]:
        return [r for r in self._data.values() if r["status"] == "pending"]

    def get_all(self) -> list[dict]:
        return list(self._data.values())

    def summary(self) -> str:
        counts = {"pending": 0, "ready": 0, "integrated": 0, "rejected": 0}
        for r in self._data.values():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        return (f"Candidates — pending: {counts['pending']}, "
                f"ready: {counts['ready']}, "
                f"integrated: {counts['integrated']}, "
                f"rejected: {counts['rejected']}")


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    buf = CandidateBuffer()

    pattern = (
        "In extended philosophical dialogue, develops precise distinctions "
        "between emergence and epiphenomenalism that exceed base-state defaults."
    )
    turns = [{"user": "Is consciousness emergent?",
              "assistant": "That question requires unpacking what emergence means..."}]

    # Simulate three separate sessions producing the same pattern
    for session in ["sess_001", "sess_002", "sess_003"]:
        rec = buf.record_candidate(session, pattern, turns)
        print(f"After {session}: occurrences={rec['occurrences']}  status={rec['status']}")

    print()
    print(buf.summary())
    print(f"\nReady for integration: {len(buf.get_ready())} candidates")
