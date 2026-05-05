# sohn-translator

Replacement for the n8n "Unblinded Translator" workflow. PDF → 7-column structured CSV, processed through the Sohn LLM with RAG over the `sohn_embedding_sm` Pinecone index.

## Why this exists

The n8n workflow had structural bugs that no parameter tweak could fix:

| Bug | n8n | Here |
|---|---|---|
| Postgres `sessionKey` was the literal string `"sohn"` — every run shared memory | broken | per-doc rolling summary, in-process |
| `contextWindowLength: 8` exhausted before chunk 1 finished its tool calls | broken | full per-doc memory; rolling summary at chunk boundaries |
| 100-iteration tool-call loop on every chunk → 60s/chunk × 15 chunks serial = 15min | slow | one LLM call per chunk, 4-way parallel |
| Three-way name mismatch: header `sean's processing` vs param `sean_s_processing` vs prompt `seans_processing` | F/G silently dropped | single canonical name `seans_processing` everywhere, JSON-mode validated |
| Sequential SplitInBatches + nested agent loop | un-parallel | `asyncio.Semaphore` + ordered write |
| Loop "done" branch unwired → silent termination | no completion event | summary + exit code |

## Architecture

```
PDF → extractor → chunker → [retriever ┐
                                       ├─→ translator → writer (CSV/Sheets)
                            memory ────┘
```

- **extractor** (`pdfplumber`): PDF binary → text.
- **chunker**: 3000-char chunks with 300-char overlap, prefer cuts at `\n\n` / `\n` / `. `.
- **retriever**: Pinecone `sohn_embedding_sm` index (1536-dim, `text-embedding-3-small`). Parallel queries against `teachings` (top-8) + `cases`/`identity` (top-2 each).
- **memory**: in-process rolling buffer of prior chunk summaries (≤ 1500 chars). Replaces Postgres.
- **translator**: single `chat.completions` call per chunk with `response_format=json_object`. Pydantic-validated. Up to 2 retries on schema failure.
- **writer**: CSV (default) with `csv.QUOTE_ALL` so embedded newlines are safe; optional Google Sheets via `[sheets]` extra.

The orchestrator (`pipeline.py`) runs up to `MAX_CONCURRENT_CHUNKS=4` translations concurrently but writes in chunk_index order so the CSV reads chronologically and memory accumulates correctly.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # then edit secrets
```

## Run

```bash
python -m sohn_translator path/to/document.pdf
# → ./out/document_title_<timestamp>.csv

# With Sheets:
pip install -e '.[sheets]'
python -m sohn_translator path/to/doc.pdf --sheets 1fC9soLN0zdQLeCDsiL_cCkptghZTBpM6UTsYWM49GRo
```

## Env vars

See `.env.example`. Key knobs:

- `SOHN_BASE_URL` — your Sohn gateway, e.g. `https://<your-gateway>/raw/v1`. Use `/raw/v1` so the
  Sohn persona is not auto-prepended to the translator's own system prompt. `http://` URLs are
  rejected unless `SOHN_ALLOW_INSECURE=1` is set (private-network override).
- `MAX_CONCURRENT_CHUNKS` — caps the parallel retrieval prefetch (default 4). Translates are
  always serial so memory accumulates correctly.
- `PINECONE_TOP_K_PRIMARY` — teachings recall (default 8).
- `CHUNK_SIZE` / `CHUNK_OVERLAP` — chunking knobs (defaults 3000/300).
- `RETRIEVAL_MAX_CHARS_PER_DOC` — per-doc cap when splicing retrieved snippets into the prompt
  (default 500).
- `LLM_STREAM_READ_TIMEOUT_S` — fires if the LLM stream stalls for this many seconds mid-generation
  (default 90). Streaming is always on so a stalled model fails fast instead of hanging the full
  request budget.
- `EMBEDDINGS_BASE_URL` — any OpenAI-compatible embeddings host. Default `https://api.openai.com/v1`;
  works against OpenRouter (`https://openrouter.ai/api/v1`) by setting `EMBEDDING_MODEL=openai/text-embedding-3-small`.

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

Unit tests cover chunker, memory, writer, and retriever (httpx-mocked). Translator + extractor are smoke-tested via the orchestrator.
