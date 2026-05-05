"""Output writers: pluggable sinks for translated rows.

Two implementations are provided:

* :class:`CSVWriter` — writes rows to a local CSV file. Always available.
* :class:`SheetsWriter` — appends rows to a Google Sheet tab. Optional;
  requires the ``sheets`` extra (``pip install sohn-translator[sheets]``).
"""
from __future__ import annotations

import asyncio
import csv
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sohn_translator.config import Config
from sohn_translator.schema import COLUMN_HEADERS, TranslatedRow

_CSV_HEADER: list[str] = ["chunk_index", *COLUMN_HEADERS]
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_ILLEGAL_SHEET_TITLE_RE = re.compile(r"[\[\]\\/?*:]+")
_MAX_FILENAME_LEN = 80
_MAX_SHEET_TITLE_LEN = 100


@runtime_checkable
class Writer(Protocol):
    async def open(self, document_title: str) -> None: ...
    async def append_rows(self, chunk_index: int, rows: list[TranslatedRow]) -> None: ...
    async def close(self) -> None: ...


def _sanitize_filename(title: str) -> str:
    lowered = title.replace("\x00", "").strip().lower()
    cleaned = _SAFE_FILENAME_RE.sub("_", lowered).strip("_")
    if not cleaned:
        cleaned = "document"
    return cleaned[:_MAX_FILENAME_LEN]


def _sanitize_sheet_title(title: str) -> str:
    no_nul = title.replace("\x00", "")
    cleaned = _ILLEGAL_SHEET_TITLE_RE.sub("", no_nul).strip()
    if not cleaned:
        cleaned = "Sheet1"
    return cleaned[:_MAX_SHEET_TITLE_LEN]


def _utc_iso_compact() -> str:
    # Includes microseconds + a short uuid to avoid collisions across rapid runs.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}Z"


def _row_to_record(chunk_index: int, row: TranslatedRow) -> list[str]:
    return [str(chunk_index), *(getattr(row, col) for col in COLUMN_HEADERS)]


class CSVWriter:
    """Append translated rows to a local CSV file. Concurrency-safe."""

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._lock = asyncio.Lock()
        self._fh = None  # type: ignore[assignment]
        self._writer: csv._writer | None = None  # type: ignore[name-defined]
        self.path: Path | None = None

    async def open(self, document_title: str) -> None:
        def _open_sync() -> tuple[Path, Any]:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_sanitize_filename(document_title)}_{_utc_iso_compact()}.csv"
            path = self._output_dir / filename
            fh = open(path, "w", newline="", encoding="utf-8")
            return path, fh

        self.path, self._fh = await asyncio.to_thread(_open_sync)
        self._writer = csv.writer(self._fh, quoting=csv.QUOTE_ALL)
        self._writer.writerow(_CSV_HEADER)
        self._fh.flush()

    async def append_rows(self, chunk_index: int, rows: list[TranslatedRow]) -> None:
        if self._writer is None or self._fh is None:
            raise RuntimeError("CSVWriter.append_rows called before open()")
        records = [_row_to_record(chunk_index, r) for r in rows]
        async with self._lock:
            self._writer.writerows(records)
            self._fh.flush()

    async def close(self) -> None:
        async with self._lock:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
            self._fh = None
            self._writer = None


class SheetsWriter:
    """Append translated rows to a Google Sheets tab. Requires `sheets` extra."""

    _RANGE_COLS = "A:H"

    def __init__(self, cfg: Config, spreadsheet_id: str) -> None:
        try:
            from googleapiclient.discovery import build  # noqa: F401
            from google.oauth2.service_account import Credentials  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "SheetsWriter requires Google client libraries. "
                "Install with: pip install sohn-translator[sheets]"
            ) from e

        creds_path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
        if not creds_path:
            raise RuntimeError(
                "Env var GOOGLE_SERVICE_ACCOUNT_JSON must point to a "
                "service account JSON file."
            )

        from googleapiclient.discovery import build
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_file(creds_path, scopes=scopes)
        self._cfg = cfg
        self._spreadsheet_id = spreadsheet_id
        self._service = build("sheets", "v4", credentials=credentials, cache_discovery=False)
        self._lock = asyncio.Lock()
        self._sheet_title: str | None = None

    async def open(self, document_title: str) -> None:
        title = _sanitize_sheet_title(document_title)
        self._sheet_title = title

        def _create_and_seed() -> None:
            body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
            self._service.spreadsheets().batchUpdate(
                spreadsheetId=self._spreadsheet_id, body=body
            ).execute()
            self._service.spreadsheets().values().update(
                spreadsheetId=self._spreadsheet_id,
                range=f"{title}!A1:H1",
                valueInputOption="RAW",
                body={"values": [_CSV_HEADER]},
            ).execute()

        await asyncio.to_thread(_create_and_seed)

    async def append_rows(self, chunk_index: int, rows: list[TranslatedRow]) -> None:
        if self._sheet_title is None:
            raise RuntimeError("SheetsWriter.append_rows called before open()")
        records = [_row_to_record(chunk_index, r) for r in rows]
        title = self._sheet_title

        def _append() -> None:
            self._service.spreadsheets().values().append(
                spreadsheetId=self._spreadsheet_id,
                range=f"{title}!{self._RANGE_COLS}",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": records},
            ).execute()

        async with self._lock:
            await asyncio.to_thread(_append)

    async def close(self) -> None:
        return None
