"""Subprocess wrapper for the Claude Code CLI as proposer.

Mirrors Stanford's reference_examples/text_classification/claude_wrapper.py
but stripped to the parts we actually use. The proposer runs as:

    claude -p <prompt>
        --model opus
        --allowed-tools <comma list>
        --strict-mcp-config
        --append-system-prompt <skill content>
        --output-format json
        --max-turns 200

Read by `meta_harness.py:propose_claude(...)`. Exits non-zero on failure.

Authentication: the wrapper deliberately POPS `ANTHROPIC_API_KEY` from the
env so the CLI uses subscription auth (avoids API rate limits). The caller
restores the key after the proposer returns.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProposerResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    wall_seconds: float = 0.0
    log_file: Path | None = None
    extra: dict = field(default_factory=dict)

    def show(self) -> None:
        print(f"[proposer] exit={self.exit_code} wall={self.wall_seconds:.1f}s")
        if self.log_file:
            print(f"[proposer] log: {self.log_file}")
        # Print last few stderr lines if non-empty (failure context).
        if self.exit_code != 0 and self.stderr:
            print(f"[proposer] stderr tail:\n{self.stderr[-1000:]}")


def run(
    *,
    prompt: str,
    model: str = "opus",
    allowed_tools: list[str] | None = None,
    skills: list[str | Path] | None = None,
    cwd: str | Path | None = None,
    log_dir: str | Path | None = None,
    name: str = "iter",
    timeout_seconds: int = 2400,
    effort: str = "max",
) -> ProposerResult:
    """Invoke `claude -p <prompt>` as a subprocess.

    `skills` is a list of skill DIRECTORIES (each containing SKILL.md). We
    concatenate their SKILL.md contents into one --append-system-prompt blob.
    """
    if allowed_tools is None:
        allowed_tools = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{name}.log") if log_dir is not None else None

    # Concatenate skill contents.
    skill_text = ""
    for s in skills or []:
        sp = Path(s)
        skill_md = sp / "SKILL.md" if sp.is_dir() else sp
        if skill_md.is_file():
            skill_text += "\n\n" + skill_md.read_text(encoding="utf-8")

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--allowed-tools", ",".join(allowed_tools),
        "--output-format", "stream-json",
        "--max-turns", "200",
        "--strict-mcp-config",
    ]
    if skill_text:
        cmd.extend(["--append-system-prompt", skill_text])
    if effort:
        # Best-effort env hint; not all CC versions read it.
        os.environ["CLAUDE_EFFORT"] = effort

    # Subscription auth — drop the API key.
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("CLAUDECODE", None)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return ProposerResult(
            exit_code=124,
            stdout="",
            stderr=f"timeout after {timeout_seconds}s: {e}",
            wall_seconds=time.time() - t0,
            log_file=log_file,
        )
    except FileNotFoundError as e:
        return ProposerResult(
            exit_code=127,
            stdout="",
            stderr=f"`claude` CLI not found: {e}",
            wall_seconds=time.time() - t0,
            log_file=log_file,
        )

    wall = time.time() - t0
    if log_file is not None:
        log_file.write_text(
            f"# {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"# cmd: {' '.join(cmd[:4])} (... +{len(cmd)-4} more args)\n"
            f"# exit={result.returncode} wall={wall:.1f}s\n\n"
            f"## STDOUT\n{result.stdout}\n\n"
            f"## STDERR\n{result.stderr}\n",
            encoding="utf-8",
        )
    return ProposerResult(
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        wall_seconds=wall,
        log_file=log_file,
    )
