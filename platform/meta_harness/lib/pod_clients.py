"""Pre-configured HTTP clients for the SGLang engine + retrieval service.

Per domain_spec.md Q11 (signed off): the harness talks to the production
retrieval service over HTTP only. No vendored `acti_retrieval` submodule.

Two clients:
  - EngineClient: httpx.AsyncClient pointing at the SGLang proxy
    (default http://127.0.0.1:8080/raw — production proxy's no-persona
    endpoint). Used by Spark for /v1/chat/completions calls.
  - RetrievalClient: thin wrapper that posts to /v1/retrieve on the
    Railway-deployed retrieval service. Used by baseline_spark.py to
    capture the raw hits per call.

Both clients read credentials from env so the same code runs on the pod
without changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

# ---------- env defaults ----------

# NOTE: spark.run_agent_sync posts to "/v1/chat/completions", so the engine
# base_url MUST NOT include the /v1 suffix or you get a double //v1/v1.
# Production's gateway.py uses ENGINE_URL="http://127.0.0.1:8000" (host only);
# we use "http://127.0.0.1:8080/raw" — the proxy's no-persona passthrough.
DEFAULT_ENGINE_BASE_URL = os.environ.get(
    "ACTI_HARNESS_ENGINE_BASE_URL", "http://127.0.0.1:8080/raw"
)
DEFAULT_ENGINE_API_KEY = os.environ.get("ACTI_EVAL_API_KEY", "")
DEFAULT_RETRIEVAL_BASE_URL = os.environ.get(
    "ACTI_LIBRARY_BASE_URL", ""
).rstrip("/")
DEFAULT_RETRIEVAL_API_KEY = os.environ.get("ACTI_LIBRARY_API_KEY", "")
DEFAULT_TIMEOUT_S = float(os.environ.get("ACTI_HARNESS_TIMEOUT_S", "60"))


# ---------- engine client ----------


def make_engine_client(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float | None = None,
) -> httpx.AsyncClient:
    """Construct an httpx.AsyncClient configured for the SGLang proxy.

    The returned client has base_url set, so callers POST to /chat/completions
    rather than /v1/chat/completions — the /v1 (or /raw/v1) prefix is in
    base_url.

    Caller is responsible for closing the client (use `async with`).
    """
    bu = (base_url or DEFAULT_ENGINE_BASE_URL).rstrip("/")
    key = api_key if api_key is not None else DEFAULT_ENGINE_API_KEY
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return httpx.AsyncClient(
        base_url=bu,
        headers=headers,
        timeout=timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S,
    )


# ---------- retrieval client ----------


@dataclass(frozen=True)
class RetrieveRequest:
    """Mirrors what library.py:handle_recall_context sends."""

    query: str
    intent: str = "general"
    subject_entity: str | None = None
    case_id: str | None = None
    top_k: int = 8

    def to_payload(self) -> dict:
        out: dict = {
            "query": self.query,
            "intent": self.intent,
            "top_k": int(self.top_k),
        }
        if self.subject_entity:
            out["subject_entity"] = self.subject_entity
        if self.case_id:
            out["case_id"] = self.case_id
        return out


@dataclass(frozen=True)
class RetrieveResponse:
    """Parsed response from /v1/retrieve. `hits` is the raw list."""

    hits: list[dict]
    raw: dict
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class RetrievalClient:
    """Async client for the Unblinded retrieval service.

    Stateful only in that it holds a configured base_url + api_key and a
    long-lived httpx.AsyncClient. Reusable across many calls.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        bu = (base_url or DEFAULT_RETRIEVAL_BASE_URL).rstrip("/")
        if not bu:
            raise ValueError(
                "retrieval base_url is empty — set ACTI_LIBRARY_BASE_URL or "
                "pass base_url= explicitly"
            )
        key = api_key if api_key is not None else DEFAULT_RETRIEVAL_API_KEY
        if not key:
            raise ValueError(
                "retrieval api_key is empty — set ACTI_LIBRARY_API_KEY or "
                "pass api_key= explicitly"
            )
        self._base_url = bu
        self._timeout_s = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "RetrievalClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout_s,
        )
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        """POST /v1/retrieve and return parsed hits.

        Never raises on network errors; surfaces them in `RetrieveResponse.error`.
        """
        if self._client is None:
            raise RuntimeError(
                "RetrievalClient must be used as an async context manager"
            )
        try:
            resp = await self._client.post("/v1/retrieve", json=req.to_payload())
        except httpx.HTTPError as e:
            return RetrieveResponse(hits=[], raw={}, error=f"network: {e}")
        if resp.status_code != 200:
            return RetrieveResponse(
                hits=[],
                raw={},
                error=f"http {resp.status_code}: {resp.text[:300]}",
            )
        try:
            payload = resp.json()
        except ValueError as e:
            return RetrieveResponse(hits=[], raw={}, error=f"non-JSON: {e}")
        hits = payload.get("hits") or []
        return RetrieveResponse(hits=hits, raw=payload, error=None)
