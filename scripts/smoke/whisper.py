"""Smoke test: faster-whisper (Phase 1 local transcriber) on short audio clips."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

log = logging.getLogger("smoke.whisper")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from faster_whisper import WhisperModel

    t0 = time.perf_counter()
    model = WhisperModel(
        args.size, device=args.device, compute_type=args.compute_type, download_root="checkpoints/whisper"
    )
    load_s = time.perf_counter() - t0
    log.info("loaded %s in %.1fs", args.size, load_s)

    rows = []
    for path in args.audio:
        # audio.wav is 16 kHz mono from extract; pass samples directly (PyAV 19 breaks the built-in decoder)
        sr, pcm = wavfile.read(path)
        assert sr == 16000, f"expected 16 kHz, got {sr}"
        audio = pcm.astype(np.float32) / 32768.0
        t1 = time.perf_counter()
        segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)
        segs = [
            {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segments
        ]
        dt = time.perf_counter() - t1
        rows.append(
            {
                "audio": path,
                "language": info.language,
                "lang_prob": round(info.language_probability, 2),
                "duration_s": round(info.duration, 1),
                "seconds": round(dt, 2),
                "segments": segs,
            }
        )
        log.info(
            "%s: %s (%.2f) %.1fs audio in %.2fs | %s",
            Path(path).parent.name,
            info.language,
            info.language_probability,
            info.duration,
            dt,
            " ".join(s["text"] for s in segs)[:150],
        )

    out = {"size": args.size, "compute_type": args.compute_type, "load_s": round(load_s, 1), "runs": rows}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
