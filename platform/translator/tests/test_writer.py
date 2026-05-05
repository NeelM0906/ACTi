"""Tests for sohn_translator.writer.CSVWriter."""
from __future__ import annotations

import asyncio
import csv
import re
from pathlib import Path

import pytest

from sohn_translator.schema import COLUMN_HEADERS, TranslatedRow
from sohn_translator.writer import CSVWriter

_TIMESTAMP_RE = re.compile(r"_\d{8}T\d{6}_[0-9a-f]{6}Z\.csv$")


def _make_row(suffix: str = "") -> TranslatedRow:
    return TranslatedRow(
        topic=f"topic{suffix}",
        context=f"context{suffix}",
        formula_element=f"fe{suffix}",
        main_lesson=f"lesson{suffix}",
        solves_what_human_condition=f"solves{suffix}",
        seans_processing=f"processing{suffix}",
        seans_approach=f"approach{suffix}",
    )


@pytest.mark.unit
async def test_open_creates_file_with_sanitized_title(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("My Document: Volume #1!!")
    try:
        assert w.path is not None
        assert w.path.exists()
        name = w.path.name
        assert name.startswith("my_document_volume_1_")
        assert _TIMESTAMP_RE.search(name) is not None
    finally:
        await w.close()


@pytest.mark.unit
async def test_header_row_matches_contract(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("doc")
    await w.close()
    assert w.path is not None
    with open(w.path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == ["chunk_index", *COLUMN_HEADERS]
    assert header[0] == "chunk_index"
    assert header[1] == "topic"
    assert header[-1] == "seans_approach"


@pytest.mark.unit
async def test_append_rows_preserves_order(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("doc")
    rows_a = [_make_row("_a1"), _make_row("_a2")]
    rows_b = [_make_row("_b1")]
    await w.append_rows(1, rows_a)
    await w.append_rows(2, rows_b)
    await w.close()

    assert w.path is not None
    with open(w.path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    assert rows[0] == ["chunk_index", *COLUMN_HEADERS]
    assert rows[1][0] == "1" and rows[1][1] == "topic_a1"
    assert rows[2][0] == "1" and rows[2][1] == "topic_a2"
    assert rows[3][0] == "2" and rows[3][1] == "topic_b1"
    assert len(rows) == 4


@pytest.mark.unit
async def test_filename_includes_timestamp_suffix(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("Doc Title")
    try:
        assert w.path is not None
        assert _TIMESTAMP_RE.search(w.path.name) is not None
    finally:
        await w.close()


@pytest.mark.unit
async def test_concurrent_append_writes_all_rows_intact(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("concurrent doc")

    n_chunks = 20
    rows_per_chunk = 5

    def _embedded_value(chunk: int, idx: int) -> TranslatedRow:
        # Fields with commas, newlines and quotes that must survive QUOTE_ALL.
        return TranslatedRow(
            topic=f'chunk={chunk},row={idx}',
            context="line1\nline2\nline3",
            formula_element='has "quotes" and, comma',
            main_lesson="lesson",
            solves_what_human_condition="x",
            seans_processing="y",
            seans_approach="z",
        )

    async def _append(chunk: int) -> None:
        rows = [_embedded_value(chunk, i) for i in range(rows_per_chunk)]
        await w.append_rows(chunk, rows)

    await asyncio.gather(*(_append(c) for c in range(1, n_chunks + 1)))
    await w.close()

    assert w.path is not None
    with open(w.path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    assert rows[0] == ["chunk_index", *COLUMN_HEADERS]
    body = rows[1:]
    assert len(body) == n_chunks * rows_per_chunk

    seen: set[tuple[str, str]] = set()
    for r in body:
        assert len(r) == len(COLUMN_HEADERS) + 1
        chunk_index = r[0]
        topic = r[1]
        assert topic.startswith(f"chunk={chunk_index},row=")
        assert r[2] == "line1\nline2\nline3"
        assert r[3] == 'has "quotes" and, comma'
        seen.add((chunk_index, topic))

    assert len(seen) == n_chunks * rows_per_chunk


@pytest.mark.unit
async def test_append_before_open_raises(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    with pytest.raises(RuntimeError):
        await w.append_rows(1, [_make_row()])


@pytest.mark.unit
async def test_empty_title_falls_back_safely(tmp_path: Path) -> None:
    w = CSVWriter(tmp_path)
    await w.open("!!! ???")
    try:
        assert w.path is not None
        assert w.path.exists()
        assert w.path.name.startswith("document_")
    finally:
        await w.close()
