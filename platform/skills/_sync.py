"""Sync /opt/acti/skills/<name>/SKILL.md into OpenWebUI's `skill` table.

The file system is the source of truth. Skills authored on disk under
ACTI_SKILLS_DIR show up under Workspace -> Skills in OWUI; deleting the
directory makes the row disappear on the next sync. Edits made through
the OWUI UI to a sohn-platform skill will be overwritten on the next
sync — file system wins.

Re-runnable. Idempotent: if no SKILL.md content changed since the last
run, no rows are updated and updated_at stays put.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple

DEFAULT_DB = Path(os.environ.get("OWUI_DB", "/root/open-webui-data/webui.db"))
DEFAULT_SKILLS_DIR = Path(os.environ.get("ACTI_SKILLS_DIR", "/opt/acti/skills"))
ORIGIN_TAG = "sohn-platform"


def _parse_frontmatter(text: str) -> Tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    block, body = text[4:end], text[end + 5:]
    meta: dict = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip().strip("'\"")
    return meta, body


def _admin_user_id(con: sqlite3.Connection) -> str:
    row = con.execute(
        "SELECT id FROM user WHERE role='admin' ORDER BY created_at LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError(
            "no admin user in OWUI — first user must sign up before skills can sync"
        )
    return row[0]


def sync_once(db_path: Path = DEFAULT_DB, skills_dir: Path = DEFAULT_SKILLS_DIR) -> dict:
    """Walk the skills directory once, write changes to OWUI, return counts."""
    if not db_path.exists():
        return {"skipped": "owui_db_missing", "path": str(db_path)}
    if not skills_dir.is_dir():
        return {"skipped": "skills_dir_missing", "path": str(skills_dir)}

    con = sqlite3.connect(str(db_path))
    try:
        admin_id = _admin_user_id(con)
        inserted = updated = unchanged = deleted = 0
        seen: set[str] = set()
        now = int(time.time())

        for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
            raw = skill_md.read_text()
            meta, _ = _parse_frontmatter(raw)
            name = meta.get("name") or skill_md.parent.name
            description = meta.get("description", "")
            sha = hashlib.sha256(raw.encode()).hexdigest()[:16]
            meta_json = json.dumps({
                "origin": ORIGIN_TAG,
                "source": str(skill_md),
                "sha": sha,
            })
            seen.add(name)

            row = con.execute(
                "SELECT id, content, meta FROM skill WHERE name = ?", (name,)
            ).fetchone()

            if row is None:
                con.execute(
                    "INSERT INTO skill (id, user_id, name, description, content, "
                    "meta, is_active, updated_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (str(uuid.uuid4()), admin_id, name, description, raw, meta_json, now, now),
                )
                inserted += 1
            elif row[1] != raw or row[2] != meta_json:
                con.execute(
                    "UPDATE skill SET description = ?, content = ?, meta = ?, "
                    "is_active = 1, updated_at = ? WHERE id = ?",
                    (description, raw, meta_json, now, row[0]),
                )
                updated += 1
            else:
                unchanged += 1

        # Delete OWUI rows whose origin is sohn-platform but whose source file
        # is gone. Rows without origin=sohn-platform are user-authored — leave them.
        for sid, sname, smeta in con.execute(
            "SELECT id, name, meta FROM skill"
        ).fetchall():
            try:
                m = json.loads(smeta or "{}")
            except Exception:
                m = {}
            if m.get("origin") == ORIGIN_TAG and sname not in seen:
                con.execute("DELETE FROM skill WHERE id = ?", (sid,))
                deleted += 1

        con.commit()
        return {"inserted": inserted, "updated": updated, "unchanged": unchanged, "deleted": deleted}
    finally:
        con.close()


def main() -> int:
    interval = float(os.environ.get("ACTI_SKILL_SYNC_INTERVAL", "0"))
    once = "--once" in sys.argv or interval <= 0
    while True:
        try:
            r = sync_once()
            if r.get("inserted") or r.get("updated") or r.get("deleted"):
                print(f"[skill-sync] {r}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[skill-sync] ERROR: {e}", flush=True)
        if once:
            return 0
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
