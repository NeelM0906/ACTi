"""Validation tests for the per-user memory isolation fix.

Covers:
  1. Different users get different memory dirs (no cross-leak)
  2. Same user across requests gets the same dir (memory persists)
  3. Sanitization handles weird inputs without filesystem traversal
  4. Onboarding flag flips correctly when memories appear
  5. Legacy global memory at the dir root is quarantined on startup

Run:  python test_user_isolation.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cortex
from gateway import _purge_legacy_global_memory, _resolve_user_identity


def _write_memory_file(d: Path, name: str, body: str = "user's name is X") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(
        "---\n"
        f"name: {name.removesuffix('.md')}\n"
        "description: test\n"
        "type: user\n"
        "---\n"
        + body + "\n"
    )


def test_different_users_get_different_dirs() -> None:
    base = Path(tempfile.mkdtemp())
    try:
        d_neel = cortex.user_memory_dir(base, "neel@example.com")
        d_aiko = cortex.user_memory_dir(base, "aiko@example.com")
        assert d_neel != d_aiko, "different users must get different dirs"
        assert d_neel.parent == base / "users"
        assert d_aiko.parent == base / "users"
        # Same user, twice → same dir.
        assert cortex.user_memory_dir(base, "neel@example.com") == d_neel
        print("  PASS: different users get different dirs, same user is stable")
    finally:
        shutil.rmtree(base)


def test_no_cross_user_injection() -> None:
    """The smoking gun: Neel's memory must not appear in Aiko's prompt."""
    base = Path(tempfile.mkdtemp())
    try:
        d_neel = cortex.user_memory_dir(base, "neel@example.com")
        _write_memory_file(d_neel, "user_name.md", "user's name is Neel")
        # Also write the index, since inject_memories reads MEMORY.md.
        d_neel.mkdir(parents=True, exist_ok=True)
        (d_neel / "MEMORY.md").write_text("- [User name](user_name.md) — Neel\n")

        d_aiko = cortex.user_memory_dir(base, "aiko@example.com")
        msgs = [{"role": "system", "content": "base"}]
        out = cortex.inject_memories(msgs, d_aiko, include_onboarding=False)
        sys_content = out[0]["content"]
        assert "Neel" not in sys_content, (
            f"LEAK: Aiko's system prompt mentions Neel: {sys_content!r}"
        )
        print("  PASS: cross-user leak prevented (Aiko sees no Neel data)")

        # And Neel's own session DOES see his memory.
        out_neel = cortex.inject_memories(msgs, d_neel, include_onboarding=False)
        assert "Neel" in out_neel[0]["content"], "Neel must see his own memories"
        print("  PASS: Neel's own session still sees his memories")
    finally:
        shutil.rmtree(base)


def test_sanitize_user_id_traversal_safe() -> None:
    cases = [
        ("../../etc/passwd", False),       # path traversal
        ("/abs/path", False),              # absolute path
        ("normal_user", True),             # passes through cleanly
        ("user@example.com", False),       # has '@', falls to hash
        ("", "_anonymous"),                # empty → anonymous
        (None, "_anonymous"),              # None → anonymous
        ("a" * 200, False),                # too long, hashes
    ]
    for raw, expectation in cases:
        s = cortex.sanitize_user_id(raw)
        assert "/" not in s and ".." not in s, f"unsafe segment: {s!r} from {raw!r}"
        if expectation is True:
            assert s == raw, f"clean input {raw!r} should pass through, got {s!r}"
        elif expectation == "_anonymous":
            assert s == "_anonymous", f"empty/None should map to _anonymous, got {s!r}"
        # Confirm stability: same input twice → same output.
        assert s == cortex.sanitize_user_id(raw), "sanitize must be deterministic"
    print("  PASS: sanitize_user_id is traversal-safe and deterministic")


def test_has_any_memory_and_onboarding_flag() -> None:
    base = Path(tempfile.mkdtemp())
    try:
        d = cortex.user_memory_dir(base, "fresh-user")
        assert not cortex.has_any_memory(d), "fresh user has no memories"

        msgs = [{"role": "system", "content": "base"}]
        out = cortex.inject_memories(msgs, d, include_onboarding=True)
        assert "ask the user for their full name" in out[0]["content"], (
            "onboarding block missing for fresh user"
        )
        # No memory files yet, so no "## Memory" section either.
        assert "## Memory" not in out[0]["content"], (
            "should not have a Memory section when there are no memories"
        )
        print("  PASS: fresh user gets onboarding block, no memory section")

        _write_memory_file(d, "user_name.md", "user's name is Alex")
        (d / "MEMORY.md").write_text("- [User name](user_name.md) — Alex\n")

        assert cortex.has_any_memory(d), "user with files should report has_any_memory"
        out2 = cortex.inject_memories(msgs, d, include_onboarding=False)
        assert "Alex" in out2["content" if False else 0]["content"]  # noqa
        assert "ask the user for their full name" not in out2[0]["content"], (
            "onboarding must NOT be injected once user has memory"
        )
        print("  PASS: returning user gets memories, no onboarding")
    finally:
        shutil.rmtree(base)


def test_legacy_purge() -> None:
    base = Path(tempfile.mkdtemp())
    try:
        # Simulate legacy global memory: *.md files at the root of memory_dir.
        base.mkdir(parents=True, exist_ok=True)
        (base / "user_name.md").write_text(
            "---\nname: user_name\ndescription: leaked\ntype: user\n---\nNeel\n"
        )
        (base / "MEMORY.md").write_text("- [User name](user_name.md) — Neel\n")

        # And a per-user partition that should NOT be touched.
        d_user = cortex.user_memory_dir(base, "alex")
        d_user.mkdir(parents=True)
        (d_user / "user_name.md").write_text("---\ntype: user\n---\nAlex\n")

        deleted = _purge_legacy_global_memory(base)
        assert deleted == 2, f"expected 2 files deleted, got {deleted}"

        # Root is now clean.
        root_md = [p for p in base.iterdir() if p.is_file() and p.suffix == ".md"]
        assert root_md == [], f"root still has md files: {root_md}"

        # Per-user partition untouched.
        assert (d_user / "user_name.md").exists(), (
            "per-user partition was incorrectly deleted"
        )

        # Idempotent: second call deletes nothing.
        deleted2 = _purge_legacy_global_memory(base)
        assert deleted2 == 0, "purge must be idempotent"
        print("  PASS: legacy global memory deleted; per-user dirs preserved")
    finally:
        shutil.rmtree(base)


def test_seed_signup_name() -> None:
    base = Path(tempfile.mkdtemp())
    try:
        d = cortex.user_memory_dir(base, "fresh-aiko")

        wrote = cortex.seed_signup_name(d, "Samantha Aiko")
        assert wrote, "first seed call should write"
        assert (d / cortex.SIGNUP_NAME_MEMORY_FILENAME).exists()
        index = (d / "MEMORY.md").read_text()
        assert cortex.SIGNUP_NAME_MEMORY_FILENAME in index, "index pointer missing"

        # has_any_memory now flips to True, so the gateway will skip onboarding.
        assert cortex.has_any_memory(d), "seeded user must register has_any_memory"

        msgs = [{"role": "system", "content": "base"}]
        out = cortex.inject_memories(msgs, d, include_onboarding=False)
        sys_text = out[0]["content"]
        assert "Samantha Aiko" in sys_text, "seeded name must surface in injected prompt"

        # Idempotent: re-seeding the same name is a no-op.
        wrote2 = cortex.seed_signup_name(d, "Samantha Aiko")
        assert not wrote2, "re-seed of same name should be a no-op"

        # Empty / whitespace name is rejected.
        assert cortex.seed_signup_name(d, "  ") is False
        print("  PASS: signup name seeding writes once, surfaces in prompt, is idempotent")
    finally:
        shutil.rmtree(base)


class _FakeRequest:
    """Minimal stand-in for FastAPI Request used by _resolve_user_identity."""
    def __init__(self, headers: dict[str, str]):
        self.headers = {k.lower(): v for k, v in headers.items()}


def test_resolve_user_identity_priority() -> None:
    # 1. Header wins over body.user.
    req = _FakeRequest({
        "X-OpenWebUI-User-Id": "uuid-aiko",
        "X-OpenWebUI-User-Name": "Samantha%20Aiko",
    })
    body = {"user": "different-id-from-body"}
    uid, uname = _resolve_user_identity(req, body)
    assert uid == "uuid-aiko", f"header must win, got {uid}"
    assert uname == "Samantha Aiko", f"name must be url-decoded, got {uname!r}"

    # 2. Falls back to body.user when no header.
    req2 = _FakeRequest({})
    body2 = {"user": "raw-api-user"}
    uid2, uname2 = _resolve_user_identity(req2, body2)
    assert uid2 == "raw-api-user"
    assert uname2 is None

    # 3. Empty header treated as absent.
    req3 = _FakeRequest({"X-OpenWebUI-User-Id": "  ", "X-OpenWebUI-User-Name": ""})
    uid3, uname3 = _resolve_user_identity(req3, {"user": "fallback"})
    assert uid3 == "fallback"
    assert uname3 is None

    # 4. Anonymous: nothing anywhere.
    uid4, uname4 = _resolve_user_identity(_FakeRequest({}), {})
    assert uid4 is None and uname4 is None
    print("  PASS: identity resolution prefers OWUI headers, falls back cleanly")


if __name__ == "__main__":
    print("Running per-user memory isolation tests...")
    test_different_users_get_different_dirs()
    test_no_cross_user_injection()
    test_sanitize_user_id_traversal_safe()
    test_has_any_memory_and_onboarding_flag()
    test_legacy_purge()
    test_seed_signup_name()
    test_resolve_user_identity_priority()
    print("\nAll tests passed.")
