"""Runtime configuration. All knobs live here. No magic numbers in other modules."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _env(key: str, default: str | None = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise RuntimeError(f"Required env var {key} is not set")
    return val


def _redact(s: str | None) -> str:
    if not s:
        return "''"
    return f"'{s[:4]}…<{len(s)} chars>'"


@dataclass(frozen=True)
class Config:
    # --- Sohn LLM (OpenAI-compatible gateway) ---
    sohn_base_url: str
    sohn_api_key: str
    sohn_model: str  # e.g. "Sohn"
    sohn_max_tokens: int

    # --- Embeddings (any OpenAI-compatible provider: OpenAI direct, OpenRouter, etc.) ---
    openai_api_key: str
    embedding_model: str  # must match Pinecone index dim — 3-small for sohn_embedding_sm
    embeddings_base_url: str  # e.g. "https://api.openai.com/v1" or "https://openrouter.ai/api/v1"

    # --- Pinecone ---
    pinecone_api_key: str
    pinecone_host: str  # full host including https://
    pinecone_primary_namespace: str  # "teachings"
    pinecone_supporting_namespaces: tuple[str, ...]  # e.g. ("cases","identity")
    pinecone_top_k_primary: int
    pinecone_top_k_supporting: int

    # --- Chunking ---
    chunk_size: int
    chunk_overlap: int

    # --- Parallelism + retries ---
    max_concurrent_chunks: int
    llm_max_retries: int
    llm_request_timeout_s: float

    # --- Memory (rolling per-doc summary) ---
    memory_max_chars: int

    # --- Retrieval payload (truncate each retrieved doc before splicing into prompt) ---
    retrieval_max_chars_per_doc: int

    # --- Streaming + stall detection ---
    llm_stream_read_timeout_s: float  # fires if no bytes received for this long mid-stream

    def __repr__(self) -> str:
        return (
            "Config("
            f"sohn_base_url={self.sohn_base_url!r}, "
            f"sohn_api_key={_redact(self.sohn_api_key)}, "
            f"sohn_model={self.sohn_model!r}, "
            f"openai_api_key={_redact(self.openai_api_key)}, "
            f"embedding_model={self.embedding_model!r}, "
            f"pinecone_api_key={_redact(self.pinecone_api_key)}, "
            f"pinecone_host={self.pinecone_host!r}, "
            f"chunk_size={self.chunk_size}, max_concurrent_chunks={self.max_concurrent_chunks}"
            ")"
        )

    @classmethod
    def from_env(cls) -> "Config":
        sohn_base_url = _env("SOHN_BASE_URL")
        if sohn_base_url.startswith("http://") and os.environ.get("SOHN_ALLOW_INSECURE") != "1":
            raise RuntimeError(
                f"SOHN_BASE_URL is plaintext HTTP ({sohn_base_url}). "
                "API key + prompts will travel unencrypted. "
                "Set SOHN_ALLOW_INSECURE=1 to override (only on a trusted private network)."
            )
        return cls(
            sohn_base_url=sohn_base_url,
            sohn_api_key=_env("SOHN_API_KEY"),
            sohn_model=_env("SOHN_MODEL", "Sohn"),
            sohn_max_tokens=int(_env("SOHN_MAX_TOKENS", "8192")),

            openai_api_key=_env("OPENAI_API_KEY"),
            embedding_model=_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            embeddings_base_url=_env("EMBEDDINGS_BASE_URL", "https://api.openai.com/v1"),

            pinecone_api_key=_env("PINECONE_API_KEY"),
            pinecone_host=_env("PINECONE_HOST"),
            pinecone_primary_namespace=_env("PINECONE_PRIMARY_NAMESPACE", "teachings"),
            pinecone_supporting_namespaces=tuple(
                ns.strip() for ns in _env(
                    "PINECONE_SUPPORTING_NAMESPACES", "cases,identity"
                ).split(",") if ns.strip()
            ),
            pinecone_top_k_primary=int(_env("PINECONE_TOP_K_PRIMARY", "8")),
            pinecone_top_k_supporting=int(_env("PINECONE_TOP_K_SUPPORTING", "2")),

            chunk_size=int(_env("CHUNK_SIZE", "3000")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "300")),

            max_concurrent_chunks=int(_env("MAX_CONCURRENT_CHUNKS", "4")),
            llm_max_retries=int(_env("LLM_MAX_RETRIES", "2")),
            llm_request_timeout_s=float(_env("LLM_REQUEST_TIMEOUT_S", "180")),

            memory_max_chars=int(_env("MEMORY_MAX_CHARS", "1500")),
            retrieval_max_chars_per_doc=int(_env("RETRIEVAL_MAX_CHARS_PER_DOC", "500")),
            llm_stream_read_timeout_s=float(_env("LLM_STREAM_READ_TIMEOUT_S", "90")),
        )
