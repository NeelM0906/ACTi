"""Translator unit tests that exercise streaming + the salvage pass.

We don't talk to the real LLM. Instead we mock `_consume_stream` (which
itself is what consumes the OpenAI SDK's async stream) to feed canned
content into the rest of the pipeline.
"""
from __future__ import annotations
from typing import Any
from unittest.mock import patch

import pytest

from sohn_translator.config import Config
from sohn_translator.schema import Chunk, Retrieval


def _make_cfg() -> Config:
    return Config(
        sohn_base_url="http://stub/v1", sohn_api_key="x", sohn_model="Sohn",
        sohn_max_tokens=512,
        openai_api_key="x", embedding_model="text-embedding-3-small",
        embeddings_base_url="https://api.openai.com/v1",
        pinecone_api_key="x", pinecone_host="https://stub",
        pinecone_primary_namespace="teachings",
        pinecone_supporting_namespaces=("cases",),
        pinecone_top_k_primary=2, pinecone_top_k_supporting=1,
        chunk_size=400, chunk_overlap=40,
        max_concurrent_chunks=1, llm_max_retries=1, llm_request_timeout_s=10.0,
        memory_max_chars=1000,
        retrieval_max_chars_per_doc=500,
        llm_stream_read_timeout_s=30.0,
    )


def _mk_chunk() -> Chunk:
    return Chunk(text="hello", chunk_index=1, total_chunks=1,
                 position_context="[Section 1 of 1]", session_id="s")


def _good_json() -> str:
    return (
        '{"rows":[{"topic":"t","context":"c","formula_element":"f",'
        '"main_lesson":"m","solves_what_human_condition":"s",'
        '"seans_processing":"sp","seans_approach":"sa"}],'
        '"chunk_summary":"x"}'
    )


@pytest.mark.unit
async def test_stream_content_accumulation_happy() -> None:
    """Translator.translate parses content streamed back from _consume_stream."""
    from sohn_translator.translator import Translator

    cfg = _make_cfg()
    t = Translator(cfg)
    try:
        async def fake_consume(stream: Any) -> tuple[str, int, int, str]:
            return _good_json(), 100, 50, "stop"

        async def fake_create(**kwargs: Any) -> object:
            assert kwargs["stream"] is True
            assert kwargs["stream_options"] == {"include_usage": True}
            return object()  # opaque; consumed by patched _consume_stream

        with patch.object(t._client.chat.completions, "create", side_effect=fake_create), \
             patch("sohn_translator.translator._consume_stream", side_effect=fake_consume):
            result = await t.translate(_mk_chunk(), Retrieval(primary=[], supporting=[]), "")
        assert len(result.rows) == 1
        assert result.rows[0].seans_approach == "sa"
    finally:
        await t.aclose()


@pytest.mark.unit
async def test_stream_stall_surfaces_as_error() -> None:
    """If the underlying consumer raises (e.g. ReadTimeout), translator does not hang."""
    import httpx
    from sohn_translator.translator import Translator

    cfg = _make_cfg()
    t = Translator(cfg)
    try:
        async def fake_consume(stream: Any) -> tuple[str, int, int, str]:
            raise httpx.ReadTimeout("simulated mid-stream stall")

        async def fake_create(**kwargs: Any) -> object:
            return object()

        with patch.object(t._client.chat.completions, "create", side_effect=fake_create), \
             patch("sohn_translator.translator._consume_stream", side_effect=fake_consume):
            with pytest.raises(httpx.ReadTimeout):
                await t.translate(_mk_chunk(), Retrieval(primary=[], supporting=[]), "")
    finally:
        await t.aclose()


@pytest.mark.unit
async def test_stream_with_markdown_fences_is_stripped() -> None:
    """If the model wraps JSON in ```json fences, the strip pass + parse still wins."""
    from sohn_translator.translator import Translator

    cfg = _make_cfg()
    t = Translator(cfg)
    try:
        async def fake_consume(stream: Any) -> tuple[str, int, int, str]:
            return f"```json\n{_good_json()}\n```", 100, 50, "stop"

        async def fake_create(**kwargs: Any) -> object:
            return object()

        with patch.object(t._client.chat.completions, "create", side_effect=fake_create), \
             patch("sohn_translator.translator._consume_stream", side_effect=fake_consume):
            result = await t.translate(_mk_chunk(), Retrieval(primary=[], supporting=[]), "")
        assert len(result.rows) == 1
    finally:
        await t.aclose()
