"""Skill library — file-system-backed catalog of model-callable skills.

Skills live as `<skill_name>/SKILL.md` under SKILLS_DIR. Each SKILL.md
opens with YAML-ish front matter (`name:`, `description:`) followed by
the skill body — model-facing instructions for the skill.

PUBLIC API (consumed by gateway.py and Spark):

    LOAD_SKILL_TOOL                — OpenAI-shape tool schema for the model
    load_skills() -> dict          — full discovery scan of SKILLS_DIR
    skills_manifest_block(skills)  — short manifest injected into the system prompt
    skills_dir_mtime() -> float    — for cheap hot-reload checks
    maybe_reload_skills(state)     — hot-reload if any SKILL.md changed
    handle_load_skill(args, skills) -> str
        Spark tool handler for `load_skill`. Returns the skill body or an
        error string starting with 'ERROR:'.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SKILLS_DIR = Path(os.environ.get("ACTI_SKILLS_DIR", "/opt/acti/skills"))


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------- tool schema (registered with Spark) ----------

LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "Load the full instructions of a skill from Sohn's skill library. "
            "Call this when the user's request matches a skill listed in <available_skills>. "
            "The skill body is returned as the tool result; follow it for the rest of this turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The skill name, e.g. design-md."},
            },
            "required": ["name"],
        },
    },
}


# ---------- frontmatter + discovery ----------

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Minimal `---`-fenced front matter parser. Returns (meta_dict, body)."""
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
        meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


def load_skills() -> dict[str, dict]:
    """Discover SKILL.md files under SKILLS_DIR. Returns {name: {description, body, path}}."""
    skills: dict[str, dict] = {}
    if not SKILLS_DIR.is_dir():
        return skills
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        try:
            meta, body = _parse_frontmatter(skill_md.read_text())
        except Exception as e:  # noqa: BLE001
            _log(f"[skills] {skill_md} unreadable: {e}")
            continue
        name = meta.get("name")
        if not name:
            _log(f"[skills] {skill_md} has no `name:` in front matter — skipped")
            continue
        skills[name] = {
            "name": name,
            "description": meta.get("description", ""),
            "body": body.strip(),
            "path": str(skill_md),
        }
    return skills


def skills_dir_mtime() -> float:
    """Latest mtime across SKILLS_DIR and every SKILL.md under it. Used for hot-reload."""
    if not SKILLS_DIR.is_dir():
        return 0.0
    try:
        latest = SKILLS_DIR.stat().st_mtime
    except FileNotFoundError:
        return 0.0
    for p in SKILLS_DIR.glob("*/SKILL.md"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except FileNotFoundError:
            continue
    return latest


def maybe_reload_skills(state) -> None:
    """Re-scan SKILLS_DIR if any SKILL.md changed since the last load.

    Mutates `state.skills`, `state.skills_manifest`, `state.skills_mtime`.
    Cheap when nothing changed (one stat per file).
    """
    latest = skills_dir_mtime()
    if latest > getattr(state, "skills_mtime", 0.0):
        state.skills = load_skills()
        state.skills_manifest = skills_manifest_block(state.skills)
        state.skills_mtime = latest
        _log(f"[skills] hot-reloaded: {list(state.skills.keys()) or 'none'}")


def skills_manifest_block(skills: dict[str, dict]) -> str:
    """Short manifest injected into the system prompt for skill discovery."""
    if not skills:
        return ""
    lines = [
        "<available_skills>",
        "Sohn has a skill library. Each entry below names a skill and its activation criteria. "
        "Call the `load_skill` tool with the skill name to retrieve its full instructions; the "
        "instructions are appended to the conversation as a tool result and you must follow them "
        "for the remainder of the turn. Activate at most one skill per request, and only when the "
        "description clearly matches the user's intent.",
        "",
    ]
    for name, info in skills.items():
        lines.append(f"- **{name}**: {info['description']}")
    lines.append("</available_skills>")
    return "\n".join(lines)


# ---------- tool handler (Spark calls this) ----------

async def handle_load_skill(args: dict, skills: dict) -> str:
    """Spark tool handler for `load_skill`. Returns the skill body, or
    an error string starting with 'ERROR:'.
    """
    name = args.get("name", "")
    sk = skills.get(name)
    if sk:
        return sk["body"]
    return (
        f"ERROR: skill '{name}' is not in the library. "
        f"Known: {sorted(skills.keys())}. Continue without it."
    )
