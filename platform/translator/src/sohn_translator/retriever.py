"""Async Pinecone retriever with parallel namespace fan-out + OpenAI embeddings."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import Config
from .schema import Retrieval, RetrievedDoc

log = logging.getLogger(__name__)

_PINECONE_API_VERSION = "2025-04"
_DEFAULT_TIMEOUT_S = 30.0
_RETRY_ATTEMPTS = 3
_RETRY_WAIT_BASE_S = 1.0


class RetrievalError(RuntimeError):
    """Raised when the retrieval pipeline fails (embeddings or Pinecone)."""


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPError):
        return True
    if isinstance(exc, RetrievalError) and getattr(exc, "status_code", None):
        code = exc.status_code  # type: ignore[attr-defined]
        return isinstance(code, int) and 500 <= code < 600
    return False


def _retrying() -> AsyncRetrying:
    return AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=_RETRY_WAIT_BASE_S, min=_RETRY_WAIT_BASE_S),
        retry=retry_if_exception(_is_retryable),
    )


class PineconeRetriever:
    """Embeds queries via OpenAI and fans out parallel Pinecone namespace queries."""

    def __init__(
        self, cfg: Config, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._cfg = cfg
        self._owns_client = http_client is None
        self._http: httpx.AsyncClient = http_client or httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT_S
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def embed(self, query: str) -> list[float]:
        # Configurable host so OpenAI-compatible providers (OpenRouter, Together,
        # Azure, self-hosted) work without code changes. Default OpenAI direct.
        base = self._cfg.embeddings_base_url.rstrip("/")
        url = f"{base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._cfg.openai_api_key}",
            "Content-Type": "application/json",
        }
        body = {"input": query, "model": self._cfg.embedding_model}

        async def _call() -> list[float]:
            try:
                resp = await self._http.post(url, headers=headers, json=body)
            except httpx.HTTPError:
                raise
            if resp.status_code >= 400:
                # Don't include resp.text in the exception message: OpenAI 4xx
                # responses sometimes echo a partial API key in the body. Body
                # goes to debug log only.
                log.debug("OpenAI 4xx body (truncated): %s", resp.text[:500])
                err = RetrievalError(
                    f"OpenAI embeddings HTTP {resp.status_code}"
                )
                err.status_code = resp.status_code  # type: ignore[attr-defined]
                raise err
            payload = resp.json()
            try:
                vec = payload["data"][0]["embedding"]
            except (KeyError, IndexError, TypeError) as ex:
                raise RetrievalError(f"Malformed embeddings response: {ex}") from ex
            if not isinstance(vec, list):
                raise RetrievalError("Embedding payload is not a list of floats")
            return [float(x) for x in vec]

        try:
            async for attempt in _retrying():
                with attempt:
                    return await _call()
        except RetryError as ex:
            raise RetrievalError(f"Embeddings retries exhausted: {ex}") from ex
        raise RetrievalError("Embeddings call did not return")

    async def _query_namespace(
        self, vector: list[float], namespace: str, top_k: int
    ) -> list[RetrievedDoc]:
        url = f"{self._cfg.pinecone_host.rstrip('/')}/query"
        headers = {
            "Api-Key": self._cfg.pinecone_api_key,
            "X-Pinecone-API-Version": _PINECONE_API_VERSION,
            "Content-Type": "application/json",
        }
        body = {
            "vector": vector,
            "topK": top_k,
            "namespace": namespace,
            "includeMetadata": True,
        }

        async def _call() -> list[RetrievedDoc]:
            try:
                resp = await self._http.post(url, headers=headers, json=body)
            except httpx.HTTPError:
                raise
            if resp.status_code >= 400:
                log.debug(
                    "Pinecone 4xx body ns=%s (truncated): %s",
                    namespace, resp.text[:500],
                )
                err = RetrievalError(
                    f"Pinecone query HTTP {resp.status_code} (ns={namespace})"
                )
                err.status_code = resp.status_code  # type: ignore[attr-defined]
                raise err
            payload = resp.json()
            return _matches_to_docs(payload, namespace)

        try:
            async for attempt in _retrying():
                with attempt:
                    return await _call()
        except RetryError as ex:
            raise RetrievalError(
                f"Pinecone retries exhausted (ns={namespace}): {ex}"
            ) from ex
        raise RetrievalError(f"Pinecone call did not return (ns={namespace})")

    async def retrieve(self, query_text: str) -> Retrieval:
        start = time.perf_counter()
        vector = await self.embed(query_text)

        primary_task = self._query_namespace(
            vector,
            self._cfg.pinecone_primary_namespace,
            self._cfg.pinecone_top_k_primary,
        )
        supporting_tasks = [
            self._query_namespace(vector, ns, self._cfg.pinecone_top_k_supporting)
            for ns in self._cfg.pinecone_supporting_namespaces
        ]

        # Primary failure is fatal. Supporting namespaces are best-effort: if a
        # supporting namespace 5xxs after retries, drop it and keep the primary.
        results = await asyncio.gather(
            primary_task, *supporting_tasks, return_exceptions=True
        )
        primary_or_err = results[0]
        if isinstance(primary_or_err, BaseException):
            raise primary_or_err
        primary = primary_or_err
        supporting: list[RetrievedDoc] = []
        for ns, batch in zip(self._cfg.pinecone_supporting_namespaces, results[1:]):
            if isinstance(batch, BaseException):
                log.warning("supporting namespace %s failed, skipping: %s", ns, batch)
                continue
            supporting.extend(batch)

        latency_ms = int((time.perf_counter() - start) * 1000)
        total_hits = len(primary) + len(supporting)
        log.info(
            "retrieve done query=%r hits=%d latency_ms=%d",
            query_text[:60],
            total_hits,
            latency_ms,
        )
        return Retrieval(primary=primary, supporting=supporting)


def _matches_to_docs(payload: dict[str, Any], namespace: str) -> list[RetrievedDoc]:
    matches = payload.get("matches") or []
    docs: list[RetrievedDoc] = []
    for m in matches:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        doc_id = m.get("id") or meta.get("doc_id") or ""
        docs.append(
            RetrievedDoc(
                namespace=namespace,
                score=float(m.get("score", 0.0)),
                doc_id=str(doc_id),
                source_title=_opt_str(meta.get("source_title")),
                subject_entity=_opt_str(meta.get("subject_entity")),
                text=str(meta.get("text") or ""),
            )
        )
    return docs


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
