"""End-to-end orchestrator. Wires extractor → chunker → retriever → translator → writer.

Replaces the n8n workflow's serial agent loop with bounded async parallelism.
Order of writes is preserved per chunk_index thanks to the writer's lock and the
orchestrator's gather-then-write pattern.
"""
from __future__ import annotations
import asyncio
import logging
import time
import uuid
from pathlib import Path

from .config import Config
from .schema import Chunk, Retrieval, TranslationResult, TranslatedRow
from .extractor import extract_pdf_text
from .chunker import chunk_text
from .memory import RollingMemory
from .retriever import PineconeRetriever
from .translator import Translator
from .writer import Writer

log = logging.getLogger(__name__)


class StubRetriever:
    """No-op retriever for --dry-run: no Pinecone, no embeddings, no creds."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def retrieve(self, query_text: str) -> Retrieval:
        return Retrieval(primary=[], supporting=[])

    async def aclose(self) -> None:
        return None


class StubTranslator:
    """Deterministic stand-in for the real Translator — used in --dry-run.

    Lets us validate the full pipeline (PDF extract, chunking, real Pinecone
    retrieval, CSV writing, ordered memory) without needing the live LLM.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    async def translate(
        self, chunk: Chunk, retrieval: Retrieval, memory_block: str
    ) -> TranslationResult:
        n_primary = len(retrieval.primary)
        n_supporting = len(retrieval.supporting)
        first_words = " ".join(chunk.text.split()[:8])
        row = TranslatedRow(
            topic=f"[stub] section {chunk.chunk_index} — {first_words}",
            context=f"[stub] {chunk.position_context}; "
                    f"retrieved {n_primary} primary + {n_supporting} supporting docs",
            formula_element="[stub] PROCESS / INFLUENCE / SELF triad placeholder",
            main_lesson="[stub] dry-run: real LLM not invoked",
            solves_what_human_condition="[stub] n/a",
            seans_processing="[stub] n/a",
            seans_approach="[stub] n/a",
        )
        return TranslationResult(
            rows=[row],
            chunk_summary=f"[stub] processed section {chunk.chunk_index}/{chunk.total_chunks}",
        )


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        writer: Writer,
        *,
        retriever: PineconeRetriever | StubRetriever | None = None,
        translator: Translator | StubTranslator | None = None,
    ):
        self.cfg = cfg
        self.writer = writer
        self.retriever = retriever if retriever is not None else PineconeRetriever(cfg)
        self.translator = translator if translator is not None else Translator(cfg)
        self.memory = RollingMemory(max_chars=cfg.memory_max_chars)

    async def aclose(self) -> None:
        # Translator may be a stub without aclose; check.
        close_t = getattr(self.translator, "aclose", None)
        if callable(close_t):
            await close_t()
        await self.retriever.aclose()

    async def _bounded_retrieve(
        self, chunk: Chunk, sem: asyncio.Semaphore
    ) -> Retrieval:
        async with sem:
            return await self.retriever.retrieve(chunk.text)

    async def run(self, pdf_path: str | Path) -> dict:
        t_start = time.monotonic()
        text, title = extract_pdf_text(pdf_path)
        log.info("extracted %d chars from %s (title=%r)", len(text), pdf_path, title)

        session_id = f"unblinded_{uuid.uuid4().hex[:12]}"
        chunks = chunk_text(
            text,
            session_id=session_id,
            chunk_size=self.cfg.chunk_size,
            overlap=self.cfg.chunk_overlap,
        )
        log.info("chunked into %d sections (session=%s)", len(chunks), session_id)

        await self.writer.open(title)

        # Concurrency model:
        #   - Retrievals (cheap, stateless) are prefetched in parallel up to N at a time.
        #   - Translations (slow, stateful — depends on rolling memory of prior chunks)
        #     are awaited serially. By the time we await translate for chunk K, its
        #     retrieval is already done so the only real wait is the LLM itself.
        # This guarantees memory is correct at chunk K while still hiding retrieval
        # latency under the LLM call.
        sem = asyncio.Semaphore(self.cfg.max_concurrent_chunks)
        retrieval_tasks: list[asyncio.Task[Retrieval]] = [
            asyncio.create_task(self._bounded_retrieve(c, sem)) for c in chunks
        ]

        total_rows = 0
        try:
            for fut, chunk in zip(retrieval_tasks, chunks):
                t0 = time.monotonic()
                retrieval = await fut
                memory_block = self.memory.render()
                result = await self.translator.translate(chunk, retrieval, memory_block)
                await self.writer.append_rows(chunk.chunk_index, result.rows)
                self.memory.add(chunk.chunk_index, result.chunk_summary)
                total_rows += len(result.rows)
                log.info(
                    "chunk %d/%d done in %.1fs (%d rows)",
                    chunk.chunk_index, chunk.total_chunks,
                    time.monotonic() - t0, len(result.rows),
                )
        except BaseException:
            for t in retrieval_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*retrieval_tasks, return_exceptions=True)
            raise
        finally:
            await self.writer.close()

        elapsed = time.monotonic() - t_start
        return {
            "title": title,
            "session_id": session_id,
            "num_chunks": len(chunks),
            "num_rows": total_rows,
            "elapsed_s": round(elapsed, 1),
        }


async def run_pipeline(
    cfg: Config,
    pdf_path: str | Path,
    writer: Writer,
    *,
    dry_run: bool = False,
    no_retrieval: bool = False,
) -> dict:
    translator: Translator | StubTranslator | None = (
        StubTranslator(cfg) if dry_run else None
    )
    if dry_run or no_retrieval:
        retriever: PineconeRetriever | StubRetriever | None = StubRetriever(cfg)
    else:
        retriever = None
    pipe = Pipeline(cfg, writer, translator=translator, retriever=retriever)
    try:
        return await pipe.run(pdf_path)
    finally:
        await pipe.aclose()
