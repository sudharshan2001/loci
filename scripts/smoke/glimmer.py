"""Smoke test: Muse Glimmer via llama-server. Text, image, one tool call, two reasoning strengths.

Starts llama-server, runs the checks, stops it. Reports load time, tokens/s and GPU memory.
Reasoning strength is set in the system prompt ("Reasoning strength: <low|medium|high|xhigh>"),
per the model card.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import subprocess
import time
from pathlib import Path

import requests

log = logging.getLogger("smoke.glimmer")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_objects",
            "description": "Search the spatial memory for objects matching a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def gpu_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return int(out.split()[0])


def chat(url: str, messages: list[dict], **kw) -> tuple[dict, float]:
    body = {"messages": messages, "temperature": 1.0, "top_p": 0.95, "top_k": 64, **kw}
    t = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json(), time.perf_counter() - t


def system(strength: str) -> dict:
    return {"role": "system", "content": f"Reasoning strength: {strength}"}


def report(name: str, resp: dict, dt: float) -> dict:
    msg = resp["choices"][0]["message"]
    timings = resp.get("timings", {})
    row = {
        "test": name,
        "seconds": round(dt, 2),
        "prompt_tokens": resp["usage"]["prompt_tokens"],
        "completion_tokens": resp["usage"]["completion_tokens"],
        "gen_tok_s": round(timings.get("predicted_per_second", 0), 1),
        "prompt_tok_s": round(timings.get("prompt_per_second", 0), 1),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "content": (msg.get("content") or "")[:300],
        "tool_calls": msg.get("tool_calls"),
    }
    log.info(
        "%-14s %5.1fs gen %5.1f tok/s, %d out tokens, reasoning %d chars | %s",
        name,
        dt,
        row["gen_tok_s"],
        row["completion_tokens"],
        row["reasoning_chars"],
        row["content"][:120].replace("\n", " ") or row["tool_calls"],
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="../llama.cpp/build/bin/llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ngl", default="99")
    ap.add_argument("--ctx", default="16384")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    url = f"http://127.0.0.1:{args.port}"
    base_mb = gpu_used_mb()
    cmd = [
        args.server,
        "-m",
        args.model,
        "--mmproj",
        args.mmproj,
        "--jinja",
        "-c",
        args.ctx,
        "-ngl",
        args.ngl,
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--port",
        str(args.port),
    ]
    log.info("starting: %s", " ".join(cmd))
    logf = open(Path(args.out).with_suffix(".server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError("llama-server exited; see server log")
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)
        load_s = time.perf_counter() - t0
        loaded_mb = gpu_used_mb() - base_mb
        log.info("ready in %.1fs, GPU +%d MB", load_s, loaded_mb)

        img_b64 = base64.b64encode(Path(args.image).read_bytes()).decode()
        image_msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {
                    "type": "text",
                    "text": "List the distinct physical objects visible, as a JSON array of short "
                    "noun phrases. Return only JSON.",
                },
            ],
        }
        rows = []
        q = "In one sentence: why does a robot need metric depth to pick up a cup?"
        for strength in ["low", "high"]:
            rows.append(
                report(f"text_{strength}", *chat(url, [system(strength), {"role": "user", "content": q}]))
            )
        for strength in ["low", "high"]:
            rows.append(report(f"image_{strength}", *chat(url, [system(strength), image_msg])))
        rows.append(
            report(
                "tool_call",
                *chat(
                    url,
                    [system("low"), {"role": "user", "content": "Where did I leave my keys?"}],
                    tools=TOOLS,
                ),
            )
        )
        peak_mb = gpu_used_mb() - base_mb

        summary = {
            "model": args.model,
            "ngl": args.ngl,
            "ctx": args.ctx,
            "load_s": round(load_s, 1),
            "gpu_mb_loaded": loaded_mb,
            "gpu_mb_peak": peak_mb,
            "tests": rows,
        }
        Path(args.out).write_text(json.dumps(summary, indent=2))
        log.info("peak GPU +%d MB; wrote %s", peak_mb, args.out)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
        logf.close()


if __name__ == "__main__":
    main()
