"""Sync /var/lib/acti/memory/*.md (Cortex-managed memories) into the
acti-ui `memory` table for in-UI visibility.

Cortex (platform/proxy/cortex.py) is the source of truth: it auto-
extracts facts from conversations and writes them as Markdown files
with YAML-ish frontmatter (name, description, type). This daemon
mirrors each file into the chat UI's `memory` table as a row under the
admin user, so the team can browse / curate them via Settings → Memory.

Tagging convention:
  - Row id is deterministic: `acti_cortex_<sha16(filename)>`. Cleanup
    finds and removes rows with this prefix whose source file is gone.
  - Content is the file body (frontmatter stripped) so vector retrieval
    over the memory store reads clean text.

This is a one-way mirror (filesystem → DB). Edits made through the chat
UI to platform-managed memories will be overwritten on the next sync —
file system wins, identical model to skill sync.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("OWUI_DB", "/var/lib/acti/openwebui/webui.db"))
DEFAULT_MEMORY_DIR = Path(os.environ.get("ACTI_MEMORY_DIR", "/var/lib/acti/memory"))
ID_PREFIX = "acti_cortex_"


def _strip_frontmatter(text: str) -> str:
    """Return the body of a Markdown file, stripping `---`-fenced front
    matter if present. Cortex memory files always have frontmatter, but
    handle the absent case gracefully.
    """
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + 5:].lstrip()


def _admin_user_id(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT id FROM user WHERE role='admin' ORDER BY created_at LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError(
            "no admin user in acti-ui — first user must sign up before memories can sync"
        )
    return row[0]


def _row_id(filename: str) -> str:
    sha = hashlib.sha256(filename.encode()).hexdigest()[:16]
    return f"{ID_PREFIX}{sha}"


def sync_once(db_path: Path = DEFAULT_DB, memory_dir: Path = DEFAULT_MEMORY_DIR) -> dict:
    """Walk the memory dir once, write changes to acti-ui, return counts."""
    if not db_path.exists():
        return {"skipped": "db_missing", "path": str(db_path)}
    if not memory_dir.is_dir():
        return {"skipped": "memory_dir_missing", "path": str(memory_dir)}

    con = sqlite3.connect(str(db_path))
    try:
        admin_id = _admin_user_id(con)
        inserted = updated = unchanged = deleted = 0
        seen_ids: set[str] = set()
        now = int(time.time())

        for md in sorted(memory_dir.glob("*.md")):
            if md.name == "MEMORY.md":
                # Skip the index file — its body is just one-line pointers,
                # not memory content.
                continue
            try:
                raw = md.read_text()
            except OSError:
                continue
            body = _strip_frontmatter(raw).rstrip()
            if not body:
                continue
            rid = _row_id(md.name)
            seen_ids.add(rid)

            row = con.execute(
                "SELECT content FROM memory WHERE id = ?", (rid,)
            ).fetchone()

            if row is None:
                con.execute(
                    "INSERT INTO memory (id, user_id, content, updated_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rid, admin_id, body, now, now),
                )
                inserted += 1
            elif row[0] != body:
                con.execute(
                    "UPDATE memory SET content = ?, updated_at = ? WHERE id = ?",
                    (body, now, rid),
                )
                updated += 1
            else:
                unchanged += 1

        # Delete platform-managed rows whose source file is gone. Rows that
        # don't carry our id prefix are user-authored — leave them alone.
        for rid in [
            r[0] for r in con.execute(
                "SELECT id FROM memory WHERE id LIKE ?", (f"{ID_PREFIX}%",)
            ).fetchall()
        ]:
            if rid not in seen_ids:
                con.execute("DELETE FROM memory WHERE id = ?", (rid,))
                deleted += 1

        con.commit()
        return {
            "inserted": inserted, "updated": updated,
            "unchanged": unchanged, "deleted": deleted,
        }
    finally:
        con.close()


def main() -> int:
    interval = float(os.environ.get("ACTI_MEMORY_SYNC_INTERVAL", "0"))
    once = "--once" in sys.argv or interval <= 0
    while True:
        try:
            r = sync_once()
            if r.get("inserted") or r.get("updated") or r.get("deleted"):
                print(f"[memory-sync] {r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[memory-sync] ERROR: {e}", flush=True)
        if once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
