"""Muse Glimmer through an OpenAI-compatible endpoint (local llama-server by default).

`glimmer_server()` starts llama-server for the stages that need it and stops it afterwards, unless one is
already answering at base_url (then it is reused and left running). Reasoning strength goes in the system
prompt, per the model card.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from loci.config import load_config, repo_path

log = logging.getLogger(__name__)

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def _cfg() -> dict[str, Any]:
    return load_config()["models"]["glimmer"]


def free_gpu_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def _healthy(base_url: str) -> bool:
    try:
        return requests.get(base_url.removesuffix("/v1") + "/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


@contextmanager
def glimmer_server(log_path: Path | None = None) -> Iterator[str]:
    """Yield the base_url of a running Glimmer server, starting a local one if needed."""
    cfg = _cfg()
    base_url = cfg["base_url"]
    if _healthy(base_url):
        log.info("using running Glimmer at %s", base_url)
        yield base_url
        return
    cmd = [
        str(repo_path(cfg["server_bin"])),
        "-m",
        str(repo_path(cfg["model"])),
        "--mmproj",
        str(repo_path(cfg["mmproj"])),
        "--jinja",
        "-c",
        str(cfg["ctx"]),
        "-ngl",
        str(cfg["n_gpu_layers"]),
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--port",
        str(cfg["port"]),
    ]
    free = free_gpu_mb()
    if free is not None and free < cfg.get("full_gpu_mb", 15200):
        # not enough room for weights + vision projector on the GPU: keep the projector on the CPU (12.4 GB);
        # images get slower (~27 s), text keeps full speed
        cmd.append("--no-mmproj-offload")
        log.warning("only %d MB GPU free: vision projector on CPU", free)
    logf = open(log_path, "w") if log_path else subprocess.DEVNULL
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    try:
        while not _healthy(base_url):
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {proc.returncode}; see {log_path}")
            time.sleep(1)
        log.info("Glimmer ready in %.0fs", time.perf_counter() - t0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_path:
            logf.close()


def image_part(path: Path) -> dict[str, Any]:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def chat(base_url: str, messages: list[dict], strength: str = "low", retries: int = 2, **kw: Any) -> dict[str, Any]:
    """One chat completion. Returns the raw message dict (content, reasoning_content, tool_calls).

    Reasoning strength goes at the top of the (single) system message. Server errors are retried: at
    temperature 1.0 the model occasionally emits a malformed tool call that llama.cpp rejects with a 500.
    """
    head = f"Reasoning strength: {strength}"
    if messages and messages[0]["role"] == "system":
        messages = [{"role": "system", "content": f"{head}\n\n{messages[0]['content']}"}, *messages[1:]]
    else:
        messages = [{"role": "system", "content": head}, *messages]
    body = {"messages": messages, **SAMPLING, **kw}
    for attempt in range(retries + 1):
        r = requests.post(f"{base_url}/chat/completions", json=body, timeout=600)
        if r.status_code < 500 or attempt == retries:
            break
        log.warning("Glimmer server error %d (attempt %d): %s", r.status_code, attempt + 1, r.text[:200])
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in text (tolerates ```json fences and surrounding prose)."""
    text = re.sub(r"```(?:json)?", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in: {text[:200]!r}")


def chat_json(base_url: str, content: list[dict] | str, strength: str = "low", retries: int = 2) -> Any:
    """Ask for JSON; on a parse failure, retry with the error appended."""
    messages: list[dict] = [{"role": "user", "content": content}]
    for attempt in range(retries + 1):
        reply = chat(base_url, messages, strength=strength)
        text = reply.get("content") or ""
        try:
            return extract_json(text)
        except ValueError as e:
            log.warning("bad JSON (attempt %d): %s", attempt + 1, e)
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "That was not valid JSON. Reply with only the JSON."},
            ]
    raise ValueError("Glimmer did not return valid JSON")
