"""Cortex prompts — verbatim ports of Anthropic Claude Code's compaction
and memory-extraction templates.

The 9-section structured summary in BASE_COMPACT_PROMPT is taken from
the leaked claude-code source (src/services/compact/prompt.ts). It has
been tuned by Anthropic against millions of agent runs; we reuse it
rather than write our own.

The continuation wrapper text is also Claude Code's exact phrasing — it
tells the model the conversation is a resumption and to continue
without asking redundant questions.

Memory extraction prompts mirror src/services/extractMemories/prompts.ts
with our 4-type taxonomy (user / feedback / project / reference).
"""
from __future__ import annotations


# =====================================================================
# COMPACTION
# =====================================================================

NO_TOOLS_PREAMBLE = """\
You are about to summarize a conversation. Output text only — do NOT \
call any tools during this summarization. Tool calls will be rejected \
and waste the turn. Produce only the summary in the format requested.\
"""


BASE_COMPACT_PROMPT = """\
Your task is to summarize the conversation above for continuation. \
Read every message — user requests, assistant replies, tool calls, tool \
results — chronologically and produce a structured summary using the 9 \
sections below.

Be specific and faithful: include filenames, exact code snippets, function \
signatures, error messages, and the user's own phrasings where relevant. \
Do not editorialize. The goal is that someone reading this summary can \
pick up the work without re-reading the conversation.

Format your output exactly as:

## Session Summary

### 1. Primary Request and Intent
What the user is ultimately trying to accomplish. One short paragraph.

### 2. Key Technical Concepts
Frameworks, libraries, protocols, file formats, design patterns referenced.

### 3. Files and Code Sections
List every file that was read, modified, or created, with the most \
relevant snippets. Format: `path/to/file.py` — what changed, with a \
short code block if it captures the essence.

### 4. Errors and Fixes
Each error encountered, root cause if known, and how it was resolved.

### 5. Problem Solving
Approaches tried, dead ends, key decisions and their rationale.

### 6. All User Messages
Compressed bullets of every user message in order. Preserve their actual \
words for short messages; paraphrase only for very long ones.

### 7. Pending Tasks
Anything explicitly requested that is not yet done.

### 8. Current Work
What was happening at the very end of the conversation, in detail. \
The next action will pick up here.

### 9. Optional Next Step
The single most natural next action, if obvious. May be empty.

Begin the summary now.\
"""


CONTINUATION_PREAMBLE = """\
This session is being continued from a previous conversation that ran \
out of context. The summary below covers the earlier portion of the \
conversation. Please continue the conversation from where we left it \
off without asking the user any further questions. Continue with the \
last task that you were asked to work on.

"""


# =====================================================================
# MEMORY EXTRACTION
# =====================================================================

MEMORY_EXTRACT_INSTRUCTIONS = """\
Analyze the most recent ~{new_message_count} messages in the conversation \
above. Decide which (if any) facts are durable and worth remembering for \
future conversations, and write them to the persistent memory directory.

You have read access to the memory directory and write access to create \
or update memory files. Do NOT interleave reads and writes across multiple \
turns; do one extraction pass and stop.

You MUST only use content from the last ~{new_message_count} messages to \
update memories. Do not invent facts.

## Memory types

There are FOUR types of memory you may write:

- **user**: who the user is, their role, project, technical background, \
  goals, preferences. Use sparingly — only save facts that change how a \
  future assistant should help them.

- **feedback**: process corrections from the user. When they explicitly \
  redirect ("don't do X", "always do Y", "the way I prefer is Z"), save \
  it as feedback with WHY (the reason given) and HOW TO APPLY (when this \
  rule kicks in).

- **project**: ongoing work context — current initiatives, goals, \
  deadlines, who is doing what. Time-bounded; expect this type to decay \
  fast as work progresses.

- **reference**: pointers to where information lives — URLs, dashboards, \
  Linear/Slack/Notion locations. Capture the URL and what it is for.

## What NOT to save

- Code patterns, conventions, architecture, file paths — derivable from \
  reading the codebase.
- Git history, recent changes — `git log` is authoritative.
- Debugging solutions, fix recipes — already in commit messages.
- Ephemeral task details, in-progress state, temporary context.
- Anything obvious from CLAUDE.md or the project README.

These exclusions apply EVEN IF the user says "save this." Ask "is this \
durable, non-obvious, and useful in a future conversation" — only save \
if all three are yes.

## How to write a memory

Each memory is a single Markdown file with frontmatter:

```markdown
---
name: <short label, e.g. user_role>
description: <one-line — what this memory covers>
type: user | feedback | project | reference
---
<body of the memory — concise, structured>
```

For feedback and project types, structure the body as:
- One-line lead with the rule or fact
- **Why:** <reason given>
- **How to apply:** <when/where this kicks in>

## Index file

After writing or updating any memory file, update `MEMORY.md` with a \
one-line pointer:

`- [Title](filename.md) — one-line hook.`

`MEMORY.md` is loaded into every conversation, so KEEP IT UNDER 200 LINES. \
Reorganize semantically by topic, not chronologically.

## Update vs. create

If a similar memory already exists, UPDATE it rather than creating a \
duplicate. Read first, then either edit or create.

Now extract memories from the recent messages.\
"""


def memory_extract_prompt(new_message_count: int) -> str:
    """Format the extraction instruction for a specific message-count window."""
    return MEMORY_EXTRACT_INSTRUCTIONS.format(new_message_count=new_message_count)
