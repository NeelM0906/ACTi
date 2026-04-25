"""End-to-end SDK test suite for the ACTi platform.

Exercises every public surface using the official OpenAI SDK exactly as a
third-party developer would. Reads SOHN_BASE_URL and SOHN_API_KEY from env.
"""
import os
import sys
import time

from openai import OpenAI, AuthenticationError

BASE_URL = os.environ.get("SOHN_BASE_URL", "http://localhost:8888/v1")
API_KEY = os.environ.get("SOHN_API_KEY", "sk-sohn-dev-key")


def hdr(s: str) -> None:
    print(f"\n{'='*70}\n  {s}\n{'='*70}")


def test_auth_rejection() -> None:
    hdr("1. Bad API key rejected with 401")
    bad = OpenAI(base_url=BASE_URL, api_key="sk-obviously-wrong")
    try:
        bad.chat.completions.create(
            model="Sohn",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        print("  FAIL: accepted a bogus key!")
        sys.exit(1)
    except AuthenticationError as e:
        print(f"  OK  rejected: {e}")


def test_models(client: OpenAI) -> None:
    hdr("2. /v1/models")
    models = client.models.list()
    for m in models.data:
        print(f"  id={m.id}  owned_by={m.owned_by}")
    assert any(m.id == "Sohn" for m in models.data)


def test_chat(client: OpenAI) -> None:
    hdr("3. Single-turn chat (thinking off)")
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="Sohn",
        messages=[{"role": "user", "content": "In one sentence, what is Zone Action?"}],
        max_tokens=400,
        temperature=0.6,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    elapsed = time.perf_counter() - t0
    print(f"  content: {(r.choices[0].message.content or '')[:300]}")
    print(f"  tokens: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
    print(f"  wall: {elapsed:.2f}s  →  {r.usage.completion_tokens/elapsed:.1f} tok/s")


def test_streaming(client: OpenAI) -> None:
    hdr("4. Streaming")
    t0 = time.perf_counter()
    first = None
    n = 0
    stream = client.chat.completions.create(
        model="Sohn",
        messages=[{"role": "user", "content": "List three priorities for Monday. Be terse."}],
        max_tokens=400,
        temperature=0.6,
        stream=True,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content or chunk.choices[0].delta.reasoning_content
        if text:
            if first is None:
                first = time.perf_counter()
                print(f"  first token at {(first - t0) * 1000:.0f}ms")
            sys.stdout.write(text)
            sys.stdout.flush()
            n += len(text)
    end = time.perf_counter()
    print()
    if first:
        print(f"  total: {(end - t0):.2f}s  ({n} chars, ~{n/(end - first):.0f} chars/s)")


def test_tool_calling(client: OpenAI) -> None:
    hdr("5. Tool calling")
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_calendar",
            "description": "Look up free time slots on the user's calendar for a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                },
                "required": ["date", "duration_minutes"],
            },
        },
    }]
    r = client.chat.completions.create(
        model="Sohn",
        messages=[{"role": "user", "content": "Find me a 30-minute slot for a call on April 25th 2026."}],
        tools=tools,
        tool_choice="auto",
        max_tokens=600,
        temperature=0.4,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    msg = r.choices[0].message
    print(f"  finish_reason: {r.choices[0].finish_reason}")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  function: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"  no tool_calls — content: {(msg.content or '')[:200]}")


def test_legacy_completions(client: OpenAI) -> None:
    hdr("6. Legacy /v1/completions")
    r = client.completions.create(
        model="Sohn",
        prompt="Give me one sharp question for my next 1:1.",
        max_tokens=400,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(f"  text: {(r.choices[0].text or '').strip()[:500]}")


def main():
    test_auth_rejection()
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    test_models(client)
    test_chat(client)
    test_streaming(client)
    test_tool_calling(client)
    test_legacy_completions(client)
    print("\n" + "=" * 70 + "\n  ALL TESTS PASSED\n" + "=" * 70)


if __name__ == "__main__":
    main()
