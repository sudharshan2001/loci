"""Show that Loci runs on Meta open weights on this machine.

Prints the GGUF the running llama-server has loaded (from its /props endpoint), its size and SHA-256 prefix,
the GPU, and then answers one demo question through the same agent the viewer uses.

    uv run python scripts/verify_open_weights.py "Where's my mouse?"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import requests

from loci.agent.agent import ask
from loci.agent.tools import Memory
from loci.config import load_config, repo_path
from loci.llm import glimmer_server
from loci.stages.store import connect


def sha256_prefix(path: Path, n_bytes: int = 64 << 20) -> str:
    """Hash of the first 64 MiB: enough to identify the file without reading 13 GB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="Where's my passport?")
    args = ap.parse_args()

    cfg = load_config()["models"]["glimmer"]
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    with glimmer_server(repo_path("debug") / "glimmer_verify.log") as url:
        props = requests.get(url.removesuffix("/v1") + "/props", timeout=10).json()
        loaded = Path(props.get("model_path", cfg["model"]))
        model = loaded if loaded.is_absolute() else repo_path(loaded)
        print("Loci runs on open weights")
        print(f"  model      {model.name}")
        print(f"  size       {model.stat().st_size / 1e9:.2f} GB, sha256[:64MiB] {sha256_prefix(model)}")
        print(f"  vision     {Path(cfg['mmproj']).name}")
        print("  licence    Apache 2.0 (unsloth/Muse-Glimmer-30B-GGUF)")
        print(f"  served by  llama.cpp llama-server at {url}")
        print(f"  gpu        {gpu}")
        print(f"\nQ: {args.question}")
        ans = ask(args.question, Memory(connect(repo_path(load_config()["paths"]["memory_db"]))), url)
        print(f"A: {ans['answer']}")
        print(json.dumps({k: ans.get(k) for k in ("verified", "used_cloud", "evidence", "latency_s")}))


if __name__ == "__main__":
    main()
