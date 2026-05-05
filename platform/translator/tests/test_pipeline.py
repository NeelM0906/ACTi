"""Pipeline integration tests using stub retriever + injected translators."""
from __future__ import annotations
import asyncio
import csv
from pathlib import Path

import pytest

from sohn_translator.config import Config
from sohn_translator.pipeline import Pipeline, StubRetriever, StubTranslator
from sohn_translator.schema import Chunk, Retrieval, TranslatedRow, TranslationResult
from sohn_translator.writer import CSVWriter


def _make_cfg() -> Config:
    return Config(
        sohn_base_url="http://stub", sohn_api_key="x", sohn_model="Sohn",
        sohn_max_tokens=512,
        openai_api_key="x", embedding_model="text-embedding-3-small",
        embeddings_base_url="https://api.openai.com/v1",
        pinecone_api_key="x", pinecone_host="https://stub",
        pinecone_primary_namespace="teachings",
        pinecone_supporting_namespaces=("cases",),
        pinecone_top_k_primary=2, pinecone_top_k_supporting=1,
        chunk_size=400, chunk_overlap=40,
        max_concurrent_chunks=3, llm_max_retries=1, llm_request_timeout_s=10.0,
        memory_max_chars=1000,
        retrieval_max_chars_per_doc=500,
        llm_stream_read_timeout_s=60.0,
    )


class _SpyTranslator:
    """Records the memory_block seen by each chunk so we can assert ordering."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def translate(
        self, chunk: Chunk, retrieval: Retrieval, memory_block: str
    ) -> TranslationResult:
        self.calls.append((chunk.chunk_index, memory_block))
        row = TranslatedRow(
            topic=f"row-{chunk.chunk_index}", context="c", formula_element="f",
            main_lesson="m", solves_what_human_condition="s",
            seans_processing="sp", seans_approach="sa",
        )
        return TranslationResult(rows=[row], chunk_summary=f"summary-{chunk.chunk_index}")

    async def aclose(self) -> None:
        return None


class _FailAtChunk:
    """Translator that succeeds for chunks < target, raises at target."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.translated: list[int] = []

    async def translate(
        self, chunk: Chunk, retrieval: Retrieval, memory_block: str
    ) -> TranslationResult:
        if chunk.chunk_index == self.fail_at:
            raise RuntimeError(f"injected failure at chunk {chunk.chunk_index}")
        self.translated.append(chunk.chunk_index)
        row = TranslatedRow(
            topic="t", context="c", formula_element="f", main_lesson="m",
            solves_what_human_condition="s", seans_processing="sp", seans_approach="sa",
        )
        return TranslationResult(rows=[row], chunk_summary=f"sum-{chunk.chunk_index}")

    async def aclose(self) -> None:
        return None


@pytest.mark.unit
async def test_happy_path_memory_is_correctly_ordered(tmp_path: Path) -> None:
    """Chunk K must see summaries for chunks 1..K-1 (and only those)."""
    cfg = _make_cfg()
    spy = _SpyTranslator()
    writer = CSVWriter(output_dir=tmp_path)
    pipe = Pipeline(cfg, writer, retriever=StubRetriever(cfg), translator=spy)

    # Build enough text for ~4 chunks at chunk_size=400
    chunks_text = (("X" * 380 + "\n\n") * 5).strip()
    pdf_path = tmp_path / "fake.pdf"

    # Bypass the PDF extractor by monkey-patching the run()'s extract step
    # via direct chunker call.
    from sohn_translator.chunker import chunk_text
    chunks = chunk_text(
        chunks_text, session_id="test", chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
    )
    assert len(chunks) >= 3

    # Drive the relevant body of run() directly so we don't need a real PDF here.
    await writer.open("happy_path")
    sem = asyncio.Semaphore(cfg.max_concurrent_chunks)
    retrieval_tasks = [
        asyncio.create_task(pipe._bounded_retrieve(c, sem)) for c in chunks
    ]
    for fut, chunk in zip(retrieval_tasks, chunks):
        retrieval = await fut
        memory_block = pipe.memory.render()
        result = await spy.translate(chunk, retrieval, memory_block)
        await writer.append_rows(chunk.chunk_index, result.rows)
        pipe.memory.add(chunk.chunk_index, result.chunk_summary)
    await writer.close()

    # Chunk 1 sees empty memory, chunk K sees summaries 1..K-1
    assert spy.calls[0][1] == ""
    for idx, mem in spy.calls[1:]:
        for k in range(1, idx):
            assert f"summary-{k}" in mem, (
                f"chunk {idx} memory missing summary-{k}: got {mem!r}"
            )

    # CSV has every chunk's row
    rows = list(csv.reader(open(writer.path)))
    assert len(rows) == len(chunks) + 1  # header + chunks


@pytest.mark.unit
async def test_translator_failure_cancels_retrieval_tasks(tmp_path: Path) -> None:
    """If translate raises mid-run, all in-flight retrievals are cancelled and writer closes."""
    cfg = _make_cfg()
    failing = _FailAtChunk(fail_at=2)
    writer = CSVWriter(output_dir=tmp_path)
    pipe = Pipeline(cfg, writer, retriever=StubRetriever(cfg), translator=failing)

    from sohn_translator.chunker import chunk_text
    chunks = chunk_text(
        "Y" * 2000, session_id="test", chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap
    )
    assert len(chunks) >= 3

    await writer.open("fail_path")
    sem = asyncio.Semaphore(cfg.max_concurrent_chunks)
    retrieval_tasks = [
        asyncio.create_task(pipe._bounded_retrieve(c, sem)) for c in chunks
    ]
    raised = False
    try:
        for fut, chunk in zip(retrieval_tasks, chunks):
            retrieval = await fut
            memory_block = pipe.memory.render()
            await failing.translate(chunk, retrieval, memory_block)
    except RuntimeError:
        raised = True
        for t in retrieval_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*retrieval_tasks, return_exceptions=True)
    finally:
        await writer.close()

    assert raised
    # No retrieval task should be left in pending state
    for t in retrieval_tasks:
        assert t.done(), "retrieval task left pending after pipeline failure"
    # Translator successfully processed only chunk 1 before failing at chunk 2
    assert failing.translated == [1]
