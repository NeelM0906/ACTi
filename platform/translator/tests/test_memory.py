"""Tests for sohn_translator.memory."""
from __future__ import annotations

import pytest

from sohn_translator.memory import RollingMemory


@pytest.mark.unit
def test_empty_memory_renders_empty_string() -> None:
    mem = RollingMemory(max_chars=500)
    assert mem.render() == ""


@pytest.mark.unit
def test_add_then_render_includes_header_and_entry() -> None:
    mem = RollingMemory(max_chars=500)
    mem.add(1, "First chunk recap.")
    out = mem.render()

    assert out.startswith("## Memory: prior chunk summaries")
    assert "[Section 1]: First chunk recap." in out


@pytest.mark.unit
def test_multiple_entries_rendered_in_chunk_index_order() -> None:
    mem = RollingMemory(max_chars=2000)
    mem.add(1, "one")
    mem.add(2, "two")
    mem.add(3, "three")
    out = mem.render()

    one = out.index("[Section 1]: one")
    two = out.index("[Section 2]: two")
    three = out.index("[Section 3]: three")
    assert one < two < three


@pytest.mark.unit
def test_eviction_drops_oldest_when_over_max_chars() -> None:
    mem = RollingMemory(max_chars=120)
    mem.add(1, "A" * 60)
    mem.add(2, "B" * 60)
    mem.add(3, "C" * 60)

    out = mem.render()
    assert len(out) <= 120
    # Newest entry must be present whole; older ones evicted from the front.
    assert "C" * 60 in out
    assert "A" * 60 not in out


@pytest.mark.unit
def test_single_oversized_entry_is_truncated_to_budget() -> None:
    mem = RollingMemory(max_chars=80)
    big = "Z" * 500
    mem.add(7, big)

    out = mem.render()
    assert len(out) <= 80
    assert out.startswith("## Memory: prior chunk summaries")
    # The only entry's payload is truncated, but its prefix should still appear.
    assert "Z" in out


@pytest.mark.unit
def test_invalid_max_chars_raises() -> None:
    with pytest.raises(ValueError):
        RollingMemory(max_chars=0)
    with pytest.raises(ValueError):
        RollingMemory(max_chars=-10)


@pytest.mark.unit
def test_render_is_idempotent_without_mutation() -> None:
    mem = RollingMemory(max_chars=300)
    mem.add(1, "alpha")
    mem.add(2, "beta")
    first = mem.render()
    second = mem.render()
    assert first == second
