"""Media generation — Lumen client + image/video tool definitions.

The ACTi proxy holds a server-side bearer token for the Lumen FastAPI
backend (running on a separate workstation, exposed via ngrok). Generated
files are mirrored from Lumen's `/api/files` endpoint into this pod's
`/var/lib/acti/media/` directory and served back to clients via nginx
under `/media/<sha>.<ext>`.

PUBLIC API (consumed by gateway.py via the Spark tool registry):

    GENERATE_IMAGE_TOOL   — OpenAI-shape tool schema for the model
    GENERATE_VIDEO_TOOL   — OpenAI-shape tool schema for the model
    media_enabled()       — bool, true iff base URL + token are set
    handle_generate_image(args) -> str
    handle_generate_video(args) -> str
        Async tool handlers for Spark. Take parsed args, return the
        string content for the resulting tool message. Never raise; on
        failure, return a string starting with 'ERROR:'.

A single asyncio.Lock serialises Lumen calls — the workstation runs on
one GPU and would queue server-side anyway. Holding the lock proxy-side
prevents a second caller from blocking inside Lumen's queue with no
progress signal.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx


# ---------- configuration ----------

LUMEN_BASE_URL = os.environ.get("ACTI_LUMEN_BASE_URL", "").rstrip("/")
LUMEN_AUTH_TOKEN = os.environ.get("ACTI_LUMEN_AUTH_TOKEN", "")
MEDIA_DIR = Path(os.environ.get("ACTI_MEDIA_DIR", "/var/lib/acti/media"))
LUMEN_IMAGE_TIMEOUT = float(os.environ.get("ACTI_LUMEN_IMAGE_TIMEOUT", "180"))
LUMEN_VIDEO_TIMEOUT = float(os.environ.get("ACTI_LUMEN_VIDEO_TIMEOUT", "600"))


def _log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------- tool schemas (registered with Spark) ----------

GENERATE_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Create one or more original images from a text description. "
            "Call this only when the user explicitly asks for an image to be drawn, generated, "
            "rendered, or visualised. Do NOT call it to fetch existing images from the web — "
            "this tool always synthesises new pixels. Generation typically takes 5–20 seconds. "
            "The tool result is a list of URLs you must reference in your final reply with "
            "standard markdown image syntax: ![short alt text](url)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What to draw. Be visual and concrete (subject, style, lighting, framing).",
                },
                "width": {"type": "integer", "description": "Pixels. Default 1024.", "default": 1024},
                "height": {"type": "integer", "description": "Pixels. Default 1024.", "default": 1024},
                "num_images": {
                    "type": "integer",
                    "description": "How many variants to generate. 1–4. Default 1.",
                    "default": 1,
                },
            },
            "required": ["prompt"],
        },
    },
}

GENERATE_VIDEO_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_video",
        "description": (
            "Create a short original video clip from a text description. "
            "Call this only when the user explicitly asks for a video / clip / animation / movie. "
            "Generation takes 30–180 seconds — use sparingly. The tool result is a single URL "
            "you must embed in your final reply with an HTML5 video tag: "
            "<video controls src=\"...\"></video>."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What the clip shows. Describe action, camera, mood, and setting.",
                },
                "duration": {
                    "type": "integer",
                    "description": "Length in seconds. 1–8. Default 4.",
                    "default": 4,
                },
                "resolution": {
                    "type": "string",
                    "enum": ["540p", "720p", "1080p"],
                    "description": "Output resolution. Default 1080p.",
                    "default": "1080p",
                },
                "aspect_ratio": {
                    "type": "string",
                    "enum": ["16:9", "9:16"],
                    "description": "Frame aspect ratio. Default 16:9.",
                    "default": "16:9",
                },
                "camera_motion": {
                    "type": "string",
                    "enum": ["none", "dolly_in", "dolly_out", "dolly_left", "dolly_right",
                             "jib_up", "jib_down", "static", "focus_shift"],
                    "description": "Optional camera move. Default none.",
                    "default": "none",
                },
            },
            "required": ["prompt"],
        },
    },
}


def media_enabled() -> bool:
    return bool(LUMEN_BASE_URL and LUMEN_AUTH_TOKEN)


_lumen_lock = asyncio.Lock()


# ---------- Lumen client ----------

async def _mirror_lumen_file(client: httpx.AsyncClient, abs_path: str) -> str | None:
    """Fetch a file from Lumen's /api/files, save under MEDIA_DIR, return public URL.

    Returns a relative URL like `/media/<sha>.<ext>` so it works regardless of
    which host the OWUI client used to reach the platform. Returns None on
    download failure.
    """
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    sha = hashlib.sha256(f"{abs_path}|{time.time_ns()}".encode()).hexdigest()[:24]
    ext = Path(abs_path).suffix or ".bin"
    fname = f"{sha}{ext}"
    dest = MEDIA_DIR / fname
    try:
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        async with client.stream(
            "GET",
            f"{LUMEN_BASE_URL}/api/files",
            params={"path": abs_path},
            headers=headers,
        ) as r:
            if r.status_code != 200:
                _log(f"[media] lumen file fetch {r.status_code} for {abs_path}")
                return None
            with dest.open("wb") as f:
                async for chunk in r.aiter_bytes():
                    f.write(chunk)
    except (httpx.HTTPError, OSError) as e:
        _log(f"[media] lumen file mirror failed: {e}")
        return None
    return f"/media/{fname}"


async def _generate_image(args: dict) -> dict:
    """Call Lumen image generation, mirror outputs, return {urls, prompt} or {error}."""
    if not media_enabled():
        return {"error": "image generation is not configured on this deployment."}
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required."}
    payload = {
        "prompt": prompt,
        "width": int(args.get("width") or 1024),
        "height": int(args.get("height") or 1024),
        "numSteps": 4,
        "numImages": max(1, min(int(args.get("num_images") or 1), 4)),
    }
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    async with _lumen_lock:
        async with httpx.AsyncClient(timeout=LUMEN_IMAGE_TIMEOUT) as c:
            try:
                resp = await c.post(
                    f"{LUMEN_BASE_URL}/api/generate-image", json=payload, headers=headers
                )
            except httpx.HTTPError as e:
                return {"error": f"upstream connection failed: {e}"}
            if resp.status_code != 200:
                return {"error": f"upstream {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            if data.get("status") == "cancelled":
                return {"error": "generation cancelled."}
            if data.get("status") != "complete":
                return {"error": f"generation status: {data.get('status', 'unknown')}"}
            paths = data.get("image_paths") or []
            if not paths:
                return {"error": "no image returned by Lumen."}
            urls: list[str] = []
            for p in paths:
                u = await _mirror_lumen_file(c, p)
                if u is None:
                    return {"error": f"failed to download generated image: {p}"}
                urls.append(u)
    return {"urls": urls, "prompt": prompt}


async def _generate_video(args: dict) -> dict:
    """Call Lumen video generation, mirror output, return {url, prompt} or {error}."""
    if not media_enabled():
        return {"error": "video generation is not configured on this deployment."}
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt is required."}
    payload = {
        "prompt": prompt,
        "resolution": args.get("resolution") or "1080p",
        "model": "fast",
        "cameraMotion": args.get("camera_motion") or "none",
        "negativePrompt": "",
        "duration": max(1, min(int(args.get("duration") or 4), 8)),
        "fps": 24,
        "audio": False,
        "imagePath": None,
        "audioPath": None,
        "aspectRatio": args.get("aspect_ratio") or "16:9",
    }
    headers = {"Authorization": f"Bearer {LUMEN_AUTH_TOKEN}"}
    async with _lumen_lock:
        async with httpx.AsyncClient(timeout=LUMEN_VIDEO_TIMEOUT) as c:
            try:
                resp = await c.post(
                    f"{LUMEN_BASE_URL}/api/generate", json=payload, headers=headers
                )
            except httpx.HTTPError as e:
                return {"error": f"upstream connection failed: {e}"}
            if resp.status_code != 200:
                return {"error": f"upstream {resp.status_code}: {resp.text[:300]}"}
            data = resp.json()
            if data.get("status") == "cancelled":
                return {"error": "generation cancelled."}
            if data.get("status") != "complete":
                return {"error": f"generation status: {data.get('status', 'unknown')}"}
            video_path = data.get("video_path")
            if not video_path:
                return {"error": "no video returned by Lumen."}
            url = await _mirror_lumen_file(c, video_path)
            if url is None:
                return {"error": "failed to download generated video."}
    return {"url": url, "prompt": prompt}


# ---------- result formatting ----------

def _format_image_result(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    urls = result.get("urls", [])
    prompt = result.get("prompt", "")
    lines = [f"Generated {len(urls)} image(s) for prompt: {prompt!r}."]
    for i, u in enumerate(urls):
        lines.append(f"- url[{i}]: {u}")
    lines.append(
        "Embed each url in your final reply using markdown image syntax: "
        "![brief alt](url). Embed each url exactly once. "
        "Do not call generate_image again."
    )
    return "\n".join(lines)


def _format_video_result(result: dict) -> str:
    if "error" in result:
        return f"ERROR: {result['error']}"
    url = result.get("url", "")
    prompt = result.get("prompt", "")
    return (
        f"Generated a video for prompt: {prompt!r}.\n"
        f"- url: {url}\n"
        f"Embed the url in your final reply using an HTML5 video element "
        f"with `controls` and a `src` attribute. Embed the url exactly once. "
        f"Do not call generate_video again."
    )


# ---------- public tool handlers (Spark calls these) ----------

async def handle_generate_image(args: dict) -> str:
    """Spark tool handler for generate_image. Returns the tool result text."""
    return _format_image_result(await _generate_image(args))


async def handle_generate_video(args: dict) -> str:
    """Spark tool handler for generate_video. Returns the tool result text."""
    return _format_video_result(await _generate_video(args))


# ---------- direct API surface (for /v1/images/generations, /v1/videos/generations) ----------

async def generate_image_raw(prompt: str, width: int, height: int, n: int) -> dict:
    """Used by /v1/images/generations. Returns the raw {urls, prompt} dict
    or {error}. Caller wraps into OpenAI shape.
    """
    return await _generate_image({
        "prompt": prompt, "width": width, "height": height, "num_images": n,
    })


async def generate_video_raw(
    prompt: str,
    duration: int = 4,
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    camera_motion: str = "none",
) -> dict:
    """Used by /v1/videos/generations. Returns {url, prompt} or {error}."""
    return await _generate_video({
        "prompt": prompt,
        "duration": duration,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "camera_motion": camera_motion,
    })
