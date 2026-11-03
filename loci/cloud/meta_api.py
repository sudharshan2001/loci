"""Meta Model API client. Every Meta call goes through here.

OpenAI-compatible chat completions at https://api.meta.ai/v1 (Muse Spark). Retries with backoff on 429 and 5xx
(honouring Retry-After), a timeout on every request, a disk cache keyed by the request body, and the key read
from MODEL_API_KEY (environment or the repo's .env). The key is never logged or printed.
Calls are refused unless cloud.enabled is true or LOCI_CLOUD=1, so --no-cloud runs make none.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import requests

from loci.config import REPO_ROOT, load_config, repo_path

log = logging.getLogger(__name__)

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"  # Standard mode: content is not used to train Meta models
MAX_ATTEMPTS = 5
TIMEOUT_S = 180

# tokens and calls in this process, for cost and speed reports
USAGE: dict[str, int] = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}


def is_meta(base_url: str | None) -> bool:
    return bool(base_url) and str(base_url).rstrip("/") == BASE_URL


def enabled() -> bool:
    return bool(load_config().get("cloud", {}).get("enabled")) or os.environ.get("LOCI_CLOUD") == "1"


def _key() -> str:
    key = os.environ.get("MODEL_API_KEY")
    env = REPO_ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY":
                key = value.strip().strip("\"'")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (environment or .env)")
    return key


def to_request(messages: list[dict], strength: str, model: str = MODEL, **kw: Any) -> dict[str, Any]:
    """The request body. Reasoning strength is a request field here, not a line in the system prompt;
    'developer' is the documented top-precedence role; top_k is not part of this API."""
    msgs = [{**m, "role": "developer"} if m["role"] == "system" else m for m in messages]
    body: dict[str, Any] = {"model": model, "messages": msgs, "reasoning_effort": strength}
    if kw.get("tools"):
        body["tools"] = kw["tools"]
    return body


def _cache_path(body: dict[str, Any]):
    h = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return repo_path(load_config()["paths"]["cache"]) / "meta_api" / f"{h}.json"


def _post(path: str, **kw: Any) -> requests.Response:
    """POST with retries on 429/5xx and network errors; raises with the status on any other failure."""
    headers = {"Authorization": f"Bearer {_key()}"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = requests.post(f"{BASE_URL}{path}", headers=headers, timeout=TIMEOUT_S, **kw)
        except requests.RequestException as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"Meta API unreachable: {type(e).__name__}") from None
            time.sleep(0.5 * 2**attempt + random.random() * 0.25)
            continue
        if (r.status_code == 429 or r.status_code >= 500) and attempt < MAX_ATTEMPTS - 1:
            wait = float(r.headers.get("Retry-After") or 0.5 * 2**attempt + random.random() * 0.25)
            log.warning("Meta API %d (attempt %d), retrying in %.1fs", r.status_code, attempt + 1, wait)
            time.sleep(wait)
            continue
        break
    if r.status_code != 200:
        raise RuntimeError(f"Meta API error {r.status_code}: {r.text[:300]}")
    return r


VOICE_MODEL = "muse-voice-transcribe-1.0"


def transcribe(wav: Path, language_bias: list[str] | None = None, keywords: list[str] | None = None) -> dict[str, Any]:
    """Muse Voice Transcribe, file mode with turns (16 kHz mono 16-bit WAV, at most 10 min / 32 MB).

    Returns the TranscribeResponse: transcript, audioDurationMs, turns [{turnId, startMs, endMs, transcript}].
    """
    if not enabled():
        raise RuntimeError("cloud is off (cloud.enabled: false); set it or LOCI_CLOUD=1 to call the Meta API")
    req: dict[str, Any] = {"model": VOICE_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"}
    if language_bias:
        req["languageBias"] = language_bias
    if keywords:
        req["keywords"] = keywords
    audio = Path(wav).read_bytes()
    h = hashlib.sha256(audio + json.dumps(req, sort_keys=True).encode()).hexdigest()
    cache = repo_path(load_config()["paths"]["cache"]) / "meta_api" / f"asr_{h}.json"
    if cache.exists():
        USAGE["cache_hits"] += 1
        return json.loads(cache.read_text())
    r = _post(
        "/asr/transcribe",
        files={
            "request": (None, json.dumps(req), "application/json"),
            "audio": (Path(wav).name, audio, "audio/wav"),
        },
    )
    USAGE["calls"] += 1
    data = r.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False))
    return data


def chat(messages: list[dict], strength: str = "low", model: str = MODEL, **kw: Any) -> dict[str, Any]:
    """One chat completion; returns choices[0].message (content, tool_calls)."""
    if not enabled():
        raise RuntimeError("cloud is off (cloud.enabled: false); set it or LOCI_CLOUD=1 to call the Meta API")
    body = to_request(messages, strength, model, **kw)
    cache = _cache_path(body)
    if cache.exists():
        USAGE["cache_hits"] += 1
        return json.loads(cache.read_text())
    data = _post("/chat/completions", json=body).json()
    usage = data.get("usage") or {}
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += usage.get("prompt_tokens") or 0
    USAGE["completion_tokens"] += usage.get("completion_tokens") or 0
    USAGE["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    msg = data["choices"][0]["message"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(msg, ensure_ascii=False))
    return msg
