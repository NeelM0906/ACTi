"""Unit tests for sohn_translator.retriever using httpx.MockTransport."""
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from sohn_translator.config import Config
from sohn_translator.retriever import PineconeRetriever, RetrievalError


def _make_config(
    primary_ns: str = "teachings",
    supporting: tuple[str, ...] = ("cases", "identity"),
    top_k_primary: int = 4,
    top_k_supporting: int = 2,
) -> Config:
    return Config(
        sohn_base_url="http://sohn.invalid/v1",
        sohn_api_key="sk-sohn",
        sohn_model="Sohn",
        sohn_max_tokens=512,
        openai_api_key="sk-openai",
        embedding_model="text-embedding-3-small",
        embeddings_base_url="https://api.openai.com/v1",
        pinecone_api_key="pc-key",
        pinecone_host="https://pinecone.invalid",
        pinecone_primary_namespace=primary_ns,
        pinecone_supporting_namespaces=supporting,
        pinecone_top_k_primary=top_k_primary,
        pinecone_top_k_supporting=top_k_supporting,
        chunk_size=3000,
        chunk_overlap=300,
        max_concurrent_chunks=2,
        llm_max_retries=1,
        llm_request_timeout_s=30.0,
        memory_max_chars=1000,
        retrieval_max_chars_per_doc=500,
        llm_stream_read_timeout_s=60.0,
    )


def _embed_response(dim: int = 1536) -> dict[str, Any]:
    return {"data": [{"embedding": [0.0] * dim, "index": 0}], "model": "x"}


def _pinecone_response(matches: list[dict[str, Any]]) -> dict[str, Any]:
    return {"matches": matches}


def _match(
    doc_id: str,
    score: float,
    *,
    text: str = "",
    source_title: str | None = None,
    subject_entity: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"text": text}
    if source_title is not None:
        metadata["source_title"] = source_title
    if subject_entity is not None:
        metadata["subject_entity"] = subject_entity
    return {"id": doc_id, "score": score, "metadata": metadata}


@pytest.mark.unit
async def test_retrieve_fans_out_to_all_namespaces_in_parallel() -> None:
    cfg = _make_config()
    captured_namespaces: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            assert request.headers["authorization"].startswith("Bearer ")
            return httpx.Response(200, json=_embed_response())

        assert request.url.host == "pinecone.invalid"
        assert request.url.path == "/query"
        assert request.headers["api-key"] == "pc-key"
        assert request.headers["x-pinecone-api-version"] == "2025-04"
        body = json.loads(request.content)
        captured_namespaces.append(body["namespace"])

        if body["namespace"] == "teachings":
            assert body["topK"] == cfg.pinecone_top_k_primary
            return httpx.Response(
                200,
                json=_pinecone_response(
                    [
                        _match(
                            "t1",
                            0.91,
                            text="primary teaching body",
                            source_title="Teaching One",
                            subject_entity="EntityA",
                        )
                    ]
                ),
            )
        assert body["topK"] == cfg.pinecone_top_k_supporting
        return httpx.Response(
            200,
            json=_pinecone_response(
                [_match(f"{body['namespace']}-doc", 0.5, text=f"{body['namespace']} body")]
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        retriever = PineconeRetriever(cfg, http_client=client)
        result = await retriever.retrieve("what is the formula?")

    assert sorted(captured_namespaces) == sorted(["teachings", "cases", "identity"])

    assert len(result.primary) == 1
    p = result.primary[0]
    assert p.namespace == "teachings"
    assert p.doc_id == "t1"
    assert p.score == pytest.approx(0.91)
    assert p.source_title == "Teaching One"
    assert p.subject_entity == "EntityA"
    assert p.text == "primary teaching body"

    assert len(result.supporting) == 2
    namespaces_in_supporting = {d.namespace for d in result.supporting}
    assert namespaces_in_supporting == {"cases", "identity"}
    for d in result.supporting:
        assert d.text.endswith(" body")


@pytest.mark.unit
async def test_retrieve_handles_missing_metadata_fields() -> None:
    cfg = _make_config(supporting=())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(200, json=_embed_response())
        # Match with empty metadata - all optional fields missing
        return httpx.Response(
            200,
            json=_pinecone_response(
                [{"id": "bare", "score": 0.1, "metadata": {}}]
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        retriever = PineconeRetriever(cfg, http_client=client)
        result = await retriever.retrieve("q")

    assert len(result.primary) == 1
    d = result.primary[0]
    assert d.doc_id == "bare"
    assert d.text == ""
    assert d.source_title is None
    assert d.subject_entity is None
    assert result.supporting == []


@pytest.mark.unit
async def test_pinecone_4xx_raises_retrieval_error() -> None:
    cfg = _make_config(supporting=())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(200, json=_embed_response())
        return httpx.Response(400, text="bad vector dim")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        retriever = PineconeRetriever(cfg, http_client=client)
        with pytest.raises(RetrievalError) as excinfo:
            await retriever.retrieve("q")

    assert "400" in str(excinfo.value)
    # Body intentionally NOT in exception message — could leak partial keys.
    assert "bad vector dim" not in str(excinfo.value)


@pytest.mark.unit
async def test_embeddings_4xx_raises_retrieval_error() -> None:
    cfg = _make_config()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(401, text="invalid api key")
        return httpx.Response(200, json=_pinecone_response([]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        retriever = PineconeRetriever(cfg, http_client=client)
        with pytest.raises(RetrievalError) as excinfo:
            await retriever.retrieve("q")

    assert "401" in str(excinfo.value)
    assert "invalid api key" not in str(excinfo.value)


@pytest.mark.unit
async def test_pinecone_5xx_retries_then_succeeds() -> None:
    cfg = _make_config(supporting=())
    pinecone_calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(200, json=_embed_response())
        pinecone_calls["n"] += 1
        if pinecone_calls["n"] < 2:
            return httpx.Response(503, text="upstream unavailable")
        return httpx.Response(
            200,
            json=_pinecone_response(
                [_match("t1", 0.5, text="ok", source_title="t")]
            ),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        retriever = PineconeRetriever(cfg, http_client=client)
        result = await retriever.retrieve("q")

    assert pinecone_calls["n"] == 2
    assert len(result.primary) == 1
    assert result.primary[0].doc_id == "t1"


@pytest.mark.unit
async def test_aclose_only_closes_owned_client() -> None:
    cfg = _make_config()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    external = httpx.AsyncClient(transport=transport)
    retriever = PineconeRetriever(cfg, http_client=external)
    await retriever.aclose()
    # External client must still be usable
    assert not external.is_closed
    await external.aclose()
