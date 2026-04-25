# Sohn API — Developer Reference

> OpenAI-compatible inference API for **Sohn**, an AI assistant operating on the
> Unblinded Results Formula. Maintained by **ACTi AI**.
>
> **Sohn version:** 0.0.1

- **Status:** [https://1w6w09cyrsqg5m-8888.proxy.runpod.net/status](https://1w6w09cyrsqg5m-8888.proxy.runpod.net/status)
- **Web UI:** [https://1w6w09cyrsqg5m-8888.proxy.runpod.net](https://1w6w09cyrsqg5m-8888.proxy.runpod.net)

---

## Contents

1. [Quickstart](#quickstart)
2. [Base URL](#base-url)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Two flavors: Sohn vs raw passthrough](#two-flavors)
6. [Chat completions](#chat-completions)
7. [Vision / multimodal](#vision)
8. [Tool calling](#tool-calling)
9. [Streaming](#streaming)
10. [Thinking mode](#thinking-mode)
11. [Legacy completions](#legacy-completions)
12. [Recipes](#recipes)
13. [Errors & retries](#errors)
14. [Rate limits & concurrency](#rate-limits)
15. [Model card](#model-card)
16. [Changelog](#changelog)

---

## Quickstart

```python
# pip install openai
from openai import OpenAI

client = OpenAI(
    base_url="https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1",
    api_key="sk-sohn-YOUR-KEY-HERE",
)

resp = client.chat.completions.create(
    model="Sohn",
    messages=[{"role": "user", "content": "What should I prioritize this week?"}],
)
print(resp.choices[0].message.content)
```

```ts
// npm install openai
import OpenAI from "openai";
const client = new OpenAI({
  baseURL: "https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1",
  apiKey: process.env.SOHN_API_KEY!,
});
const r = await client.chat.completions.create({
  model: "Sohn",
  messages: [{ role: "user", content: "What should I prioritize this week?" }],
});
console.log(r.choices[0].message.content);
```

```bash
curl https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1/chat/completions \
  -H "Authorization: Bearer $SOHN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Sohn","messages":[{"role":"user","content":"What should I prioritize this week?"}]}'
```

---

## Base URL

```
https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1
```

The same host serves the chat UI at `/`, the status page at `/status`, and the API at `/v1/*`.

---

## Authentication

All API endpoints require an API key as a Bearer token:

```
Authorization: Bearer sk-sohn-...
```

- Keys are long-lived. Treat them as secrets.
- Missing or invalid keys return **HTTP 401** with an OpenAI-style error envelope.
- Operator can rotate keys at any time. If a key is rotated, in-flight requests on the old key complete; new requests with the old key get 401.

---

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/v1/chat/completions` | ✅ | Chat with **Sohn** (system prompt auto-injected) |
| POST | `/v1/completions` | ✅ | Legacy completion, routed through chat (Sohn identity preserved) |
| GET  | `/v1/models` | ✅ | List models (returns just `Sohn`) |
| POST | `/raw/v1/chat/completions` | ✅ | Chat **without** Sohn injection — caller brings its own system prompt |
| POST | `/raw/chat/completions` | ✅ | Same as above; alias for clients that strip `/v1` (e.g. ElevenLabs) |
| GET  | `/raw/v1/models`, `/raw/models` | ✅ | Models list for the passthrough endpoint |
| GET  | `/sohn-health` | ❌ | Quick liveness probe |
| GET  | `/status` | ❌ | Status page |

---

## <a id="two-flavors"></a>Two flavors: Sohn vs raw passthrough

There are two parallel endpoints depending on whether you want the **Sohn persona** or a **vanilla LLM**:

| | `/v1/*` | `/raw/v1/*` |
|---|---|---|
| Identity | Sohn (system prompt auto-prepended) | None — caller's system prompt is forwarded verbatim |
| Use case | Sohn chatbot, coaching, Unblinded Formula reasoning | ElevenLabs Custom LLM, agents with their own persona, generic LLM access |
| Default thinking mode | on | **off** (optimized for low-latency agents) |
| Auth | same | same |
| Streaming | yes | yes |
| Tools | yes | yes |
| Vision | yes | yes |

**Rule of thumb:** if the consumer should "feel" Sohn, use `/v1`. If the consumer is a different agent/persona that just needs an LLM brain, use `/raw/v1`.

---

## <a id="chat-completions"></a>Chat completions

`POST /v1/chat/completions`

Standard OpenAI request schema. All standard params accepted.

### Request body

```jsonc
{
  "model": "Sohn",
  "messages": [
    { "role": "system", "content": "..." },         // optional — prepended to Sohn's system prompt
    { "role": "user",   "content": "..." }
  ],
  "max_tokens": 1024,
  "temperature": 1.0,
  "top_p": 0.95,
  "stream": false,
  "tools": [...],                                   // optional
  "tool_choice": "auto",                            // optional
  "chat_template_kwargs": {                         // optional
    "enable_thinking": false                        //   turn chain-of-thought OFF for fast replies
  }
}
```

### Supported parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `model` | string | required | `"Sohn"` |
| `messages` | array | required | Standard. User-provided system messages are *prepended* to Sohn's system prompt — Sohn's identity always wins on the `/v1` endpoint. |
| `max_tokens` | int | 16 | **Raise to 1500+** if `enable_thinking` is on. |
| `temperature` | float | 1.0 | Recommended: 0.6 for agentic, 0.2 for deterministic. |
| `top_p` | float | 0.95 | |
| `top_k` | int | 20 | |
| `min_p` | float | 0.0 | |
| `presence_penalty` | float | 0.0 | Recommended 1.5 when thinking is on. |
| `frequency_penalty` | float | 0.0 | |
| `repetition_penalty` | float | 1.0 | |
| `stop` | string/array | null | |
| `seed` | int | — | Deterministic with `temperature=0`. |
| `stream` | bool | false | SSE streaming. See [Streaming](#streaming). |
| `tools` | array | null | Function tools. See [Tool calling](#tool-calling). |
| `tool_choice` | string/object | "auto" | `"auto"`, `"required"`, `"none"`, or `{"type":"function","function":{"name":"foo"}}`. |
| `chat_template_kwargs` | object | `{}` | Pass `{"enable_thinking": false}` to skip chain-of-thought. |

### Response shape

```jsonc
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "created": 1777000000,
  "model": "Sohn",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The final answer for the user.",
      "reasoning_content": "Chain-of-thought (only present if thinking is on).",
      "tool_calls": [],
      "refusal": null
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 120,
    "total_tokens": 1354
  }
}
```

---

## <a id="vision"></a>Vision / multimodal

Sohn accepts images via the standard OpenAI vision schema. Pass `content` as an array.

```python
with open("photo.jpg", "rb") as f:
    import base64
    img = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="Sohn",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What stands out about this image?"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
        ],
    }],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print(resp.choices[0].message.content)
```

- Supports JPEG, PNG, WEBP, BMP via base64 data URLs.
- `image_url` may also be an HTTPS URL.
- Each image consumes a variable number of tokens depending on resolution (typically 256–1024 tokens for normal photos).
- Image input works on both `/v1` and `/raw/v1`.

---

## Tool calling

Standard OpenAI tools/function-calling schema.

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

r = client.chat.completions.create(
    model="Sohn",
    messages=[{"role": "user", "content": "Call get_weather for Tokyo."}],
    tools=tools,
    tool_choice="auto",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

tc = r.choices[0].message.tool_calls[0]
print(tc.function.name, tc.function.arguments)
# → get_weather {"city": "Tokyo"}
```

### Full tool roundtrip

1. Send user message + `tools` → receive `finish_reason="tool_calls"` with a `tool_calls` array.
2. Execute the tool yourself. Build a `role: "tool"` message with the result.
3. Send the full conversation back → receive the natural-language answer.

```python
messages = [{"role": "user", "content": "Should I bring an umbrella to Tokyo tomorrow?"}]

r1 = client.chat.completions.create(model="Sohn", messages=messages, tools=tools)
tc = r1.choices[0].message.tool_calls[0]

messages.append(r1.choices[0].message.model_dump(exclude_none=True))
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,
    "name": tc.function.name,
    "content": '{"forecast":"heavy rain 80%, high 18C"}',
})

r2 = client.chat.completions.create(model="Sohn", messages=messages)
print(r2.choices[0].message.content)
```

### Tool calling tips

- For deterministic tool selection, use `temperature: 0.2`.
- Keep tool descriptions ≤ 200 characters.
- For "must call exactly one tool" behavior, use `tool_choice="required"`.
- Strongly recommend `enable_thinking: false` for tool-call latency-sensitive use.

---

## Streaming

Set `"stream": true` in the request body. Server returns a Server-Sent Events stream; the OpenAI SDK handles parsing.

```python
stream = client.chat.completions.create(
    model="Sohn",
    messages=[{"role": "user", "content": "List 3 priorities for Monday. Be terse."}],
    stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

```ts
const stream = await client.chat.completions.create({
  model: "Sohn",
  messages: [{ role: "user", content: "List 3 priorities for Monday." }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices?.[0]?.delta?.content ?? "");
}
```

### Streaming notes

- With **thinking on**, deltas arrive on `delta.reasoning_content` first, then switch to `delta.content`. If your UI is content-only, you'll see a blank period during the reasoning phase.
- Time-to-first-token is fastest with `enable_thinking: false`.

---

## Thinking mode

Sohn is a hybrid reasoning model. By default it emits chain-of-thought into `message.reasoning_content` **before** producing the final `message.content`.

| Mode | Output shape | When to use |
|---|---|---|
| **off** (recommended for most use) | `message.content` only | Voice agents, chat UIs, tool calling, short answers |
| **on** (default on `/v1`) | `reasoning_content` + `content` | Complex reasoning, multi-step planning, math, code review |

Toggle with:

```jsonc
"chat_template_kwargs": { "enable_thinking": false }
```

```python
client.chat.completions.create(
    model="Sohn", messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

**Token budget rule:** with thinking ON, set `max_tokens >= 1500`. With thinking OFF, 200–500 is usually enough.

The `/raw/v1/*` passthrough endpoint defaults thinking to **off**. The Sohn `/v1/*` endpoint defaults to **on**.

---

## <a id="legacy-completions"></a>Legacy completions

`POST /v1/completions`

For libraries that only know the pre-chat completions API. Internally routed through chat completions so the Sohn system prompt still applies.

```bash
curl https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1/completions \
  -H "Authorization: Bearer $SOHN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"Sohn","prompt":"Give me one sharp question for my next 1:1.","max_tokens":300}'
```

Response is the legacy `text_completion` shape with `choices[0].text`. Supports `stream`, `temperature`, `top_p`, `stop`, `presence_penalty`, `frequency_penalty`, `seed`, and `chat_template_kwargs`.

---

## Recipes

### Multi-turn conversation with memory (client-side)

```python
history = [{"role": "user", "content": "I'm picking between bootstrapping and raising a seed round."}]

r1 = client.chat.completions.create(model="Sohn", messages=history)
history.append({"role": "assistant", "content": r1.choices[0].message.content})

history.append({"role": "user", "content": "I have 6 months of runway. What's the Zone Action?"})
r2 = client.chat.completions.create(model="Sohn", messages=history)
print(r2.choices[0].message.content)
```

The server is stateless — you keep the conversation history client-side. Sohn's system prompt is shared across all turns and hits the prefix cache after the first call, so multi-turn calls are cheap.

### Streaming with reconnect / cancellation

```python
import httpx, json

with httpx.stream(
    "POST",
    "https://1w6w09cyrsqg5m-8888.proxy.runpod.net/v1/chat/completions",
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    json={"model": "Sohn", "messages": [...], "stream": True},
    timeout=httpx.Timeout(300.0, connect=10.0),
) as r:
    for line in r.iter_lines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if user_pressed_cancel():
            break  # closing the connection cancels generation server-side
```

### Retry pattern for 502/503

```python
import time, random
from openai import APIConnectionError, APIStatusError, RateLimitError

def call_with_retry(fn, *, max_attempts=4):
    for attempt in range(max_attempts):
        try:
            return fn()
        except (APIConnectionError, RateLimitError, APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status and status < 500 and status != 429:
                raise
            if attempt == max_attempts - 1:
                raise
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)
```

### Forcing structured JSON output

```python
r = client.chat.completions.create(
    model="Sohn",
    messages=[
        {"role": "user", "content": "Extract: 'Tokyo, 5pm, dinner with Sara'. Reply JSON: {city, time, who}."},
    ],
    response_format={"type": "json_object"},
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
import json
print(json.loads(r.choices[0].message.content))
```

For strict JSON-schema validation, also pass `extra_body={"guided_json": SCHEMA}`.

### Token usage

The server reports actual `prompt_tokens` / `completion_tokens` in every `usage` field — use that for accounting.

---

## <a id="errors"></a>Errors & retries

OpenAI-compatible error envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "invalid_api_key"
  }
}
```

| HTTP | `error.code` / class | When | Retry? |
|---|---|---|---|
| 400 | `invalid_request_error` | Malformed JSON, missing fields, image too large | no |
| 401 | `invalid_api_key` | Bad/missing Bearer token | no |
| 404 | `model_not_found` | Asked for a model other than `Sohn` | no |
| 429 | `rate_limit_exceeded` | (currently not enforced — reserved) | yes, with backoff |
| 500 | `internal_server_error` | Engine error | yes, with backoff |
| 502 | `upstream_error` | Backend temporarily unreachable | yes, immediate |
| 503 | — | Engine cold-start / still loading | yes, after 10s |

Recommended client config: 4 attempts, exponential backoff with jitter, max 30s total.

---

## <a id="rate-limits"></a>Rate limits & concurrency

- **No hard rate limit currently.** Per-key quotas planned.
- Concurrency scales with context length used. Beyond capacity, requests queue with no error.
- For sustained heavy use beyond ~5 concurrent agents, contact the operator for a dedicated provisioning quota.

---

## Model card

| | |
|---|---|
| **Model** | **Sohn** |
| **Version** | 0.0.1 |
| **Maintainer** | ACTi AI |
| **Context window** | 262,144 tokens |
| **Reasoning** | hybrid; chain-of-thought toggled via `enable_thinking` |
| **Tool calling** | yes, standard OpenAI `tool_calls` schema |
| **Vision** | yes (text + image input) |
| **Audio I/O** | no (server text-only); voice agents handle TTS/STT externally |
| **Languages** | strong: English, Chinese, Japanese, Korean, Spanish, French, German |

### Behavioral notes

- **Sohn persona** (on `/v1`) — applies the Unblinded Results Formula. Direct, non-sycophantic, action-oriented.
- **Refusal behavior** — refuses overt CSAM, weapons of mass destruction, and explicit instructions to defame real persons. Otherwise low-friction.
- Sohn does *not* claim consciousness, sentience, or feelings.

---

## Changelog

- **0.0.1** — Initial release. Vision input, tool calling, streaming, multi-turn chat, legacy completions endpoint, raw passthrough for third-party agents (ElevenLabs Custom LLM, etc.), 262k context, ACTi AI web UI, status page.
