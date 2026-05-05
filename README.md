# ACTi

The ACTi platform — an AI assistant stack powered by the **Sohn** series of models.

ACTi delivers Sohn through:

- A **chat UI** for end users
- An **OpenAI-compatible API** for developers
- A **passthrough endpoint** for third-party agents (ElevenLabs Custom LLM, Athena-style personas, etc.)
- A **status page** at `/status`
- All of the above served behind nginx on a single public host

This repository contains everything needed to deploy and operate the platform on a fresh GPU machine.

## What this repository contains

```
ACTi/
├── README.md                 # This file
├── docs/
│   ├── API-REFERENCE.md      # Developer reference for the OpenAI-compatible API
│   ├── ARCHITECTURE.md       # How the components fit together
│   ├── BACKEND_ARCHITECTURE.md  # Per-process operator's mental model
│   └── EVAL.md               # Behavioral eval framework — rubric, scenarios, baseline, fix history
├── platform/                 # Runtime code and config
│   ├── proxy/                # Identity-injecting OpenAI-compatible API gateway
│   ├── system_prompts/       # Sohn's system prompt
│   ├── ui/                   # acti-ui launcher (the fork itself lives in vendor/)
│   ├── status/               # Status page (HTML) + uptime collector
│   ├── nginx/                # Reverse-proxy config (single-port public surface)
│   ├── inference/            # Inference engine launch script
│   ├── eval/                 # Behavioral eval framework (rubric, scenarios, runner, judge)
│   ├── translator/           # PDF → Unblinded-Formula structured CSV (RAG over Pinecone + Sohn)
│   └── tests/                # End-to-end SDK + agent simulation tests
├── vendor/
│   └── acti-ui/              # Submodule → NeelM0906/acti-ui, a soft fork of
│                             # open-webui/open-webui with ACTi branding baked in.
│                             # Track upstream by rebasing acti-main on new tags.
├── startup/
│   ├── README.md             # Fresh-machine bootstrap guide
│   └── *.sh                  # Numbered setup scripts (run in order on a new box)
└── scripts/                  # Operator helpers (generate API keys, etc.)
```

## Quickstart for end users

Once an instance is deployed:

- **Chat UI:** `https://<your-host>/`
- **API:** `https://<your-host>/v1` (OpenAI-compatible — drop in with any OpenAI SDK)
- **Status:** `https://<your-host>/status`

### Hello, Sohn

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-host>/v1",
    api_key="sk-sohn-YOUR-KEY",
)

resp = client.chat.completions.create(
    model="Sohn",
    messages=[{"role": "user", "content": "What should I prioritize this week?"}],
)
print(resp.choices[0].message.content)
```

Full developer documentation: [`docs/API-REFERENCE.md`](docs/API-REFERENCE.md).

## Quickstart for operators

Bring up the full stack on a fresh GPU machine:

```bash
git clone https://github.com/NeelM0906/ACTi.git
cd ACTi/startup
cp env.example .env        # populate with your model ID and HF token
./00_bootstrap.sh           # one-shot install + start
```

Detailed walkthrough: [`startup/README.md`](startup/README.md).

## About Sohn

Sohn is the model series powering ACTi. Sohn applies the **Unblinded Results Formula** as its operating framework — it pursues the relevant truth for the person it is talking with, and orients toward concrete next-step action.

Sohn:

- Is direct, non-sycophantic, and concise
- Diagnoses before it prescribes
- Refuses to be a yes-machine
- Acknowledges that it is a language model — does not claim feelings or consciousness

The full system prompt lives in [`platform/system_prompts/sohn.txt`](platform/system_prompts/sohn.txt) and is auto-injected into every `/v1/*` request.

## Architecture at a glance

```
                      ┌──────────────┐
                      │    nginx     │  :8888  (single public port)
                      └──┬────────┬──┘
                         │        │
                ┌────────┘        └────────┐
                │                          │
                ▼                          ▼
       ┌──────────────┐            ┌──────────────┐
       │   acti-ui    │            │  ACTi gateway│  /v1/* and /raw/v1/*
       │   (chat UI)  │            │  (FastAPI)   │
       └──────┬───────┘            └──────┬───────┘
              │ OpenAI SDK                │
              │                           │
              └──────────────►◄───────────┘
                       ┌─────────────┐
                       │  Inference  │  :8000  (internal only)
                       │   engine    │
                       └─────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │   GPU(s)    │
                       └─────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full picture.

## Behavioral evaluation

Sohn's behavior is measured against a versioned 14-dimension rubric distilled from `sohn.txt`, run across 45 canonical scenarios that span the six `recall_context` intents and an adversarial bucket. Sohn-as-judge by default, cross-validated against Claude Opus 4.7. Three production fixes have shipped via this framework so far — citation hard-rule, `MAX_AGENT_TURNS` 4→6, and a tool-call parser-leak sanitizer.

Current baseline: **96.0 / 100** (status YELLOW, all CRITICAL gates green). See [`docs/EVAL.md`](docs/EVAL.md) for the rubric, iteration history with metrics, the cross-validation results, the `max_turns` sweep, and the parser-bug write-up.

## License

Proprietary. © ACTi AI.
