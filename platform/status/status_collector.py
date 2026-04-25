"""Status collector: probes /sohn-health every 60s, writes a 24h rolling JSON file
served as a static asset by nginx for the status page to render history bars."""
import json
import os
import time
from pathlib import Path

import httpx

PROBE_URL = os.environ.get("ACTI_HEALTH_URL", "http://127.0.0.1:8080/health")
HISTORY_PATH = Path(os.environ.get("ACTI_STATUS_HISTORY_PATH", "/usr/share/nginx/html/acti-status/status_history.json"))
WINDOW_MS = 24 * 60 * 60 * 1000   # 24 hours
INTERVAL_S = 60


def load() -> list:
    if not HISTORY_PATH.exists():
        return []
    try:
        return json.loads(HISTORY_PATH.read_text()).get("points", [])
    except Exception:
        return []


def save(points: list) -> None:
    HISTORY_PATH.write_text(
        json.dumps({"points": points, "updated": int(time.time() * 1000)}, separators=(",", ":"))
    )


def main() -> None:
    points = load()
    print(f"[status_collector] starting; loaded {len(points)} historical points", flush=True)
    while True:
        t0 = time.perf_counter()
        ok = False
        try:
            r = httpx.get(PROBE_URL, timeout=10.0)
            ok = r.status_code == 200
        except Exception:
            ok = False
        ms = int((time.perf_counter() - t0) * 1000)
        now_ms = int(time.time() * 1000)
        points.append({"ts": now_ms, "ok": ok, "ms": ms})
        # prune
        cutoff = now_ms - WINDOW_MS
        points = [p for p in points if p["ts"] > cutoff]
        save(points)
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
