"""Smoke test: SAM 3.1 multiplex video predictor on a keyframe folder with text prompts.

One prompt per pass (the session is reset between prompts, as the official example requires).
Built through loci.sam3_compat (use_fa3=False, init_state kwarg fix).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("smoke.sam3_video")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--frames", required=True, help="folder of <index>.jpg keyframes")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from loci.sam3_compat import build_video_predictor

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    predictor = build_video_predictor(args.checkpoint)
    load_s = time.perf_counter() - t0
    log.info("loaded in %.1fs", load_s)

    frames = sorted(Path(args.frames).glob("*.jpg"))
    torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    session_id = predictor.handle_request({"type": "start_session", "resource_path": args.frames})[
        "session_id"
    ]
    start_s = time.perf_counter() - t1

    rows = []
    for prompt in args.prompts:
        predictor.handle_request({"type": "reset_session", "session_id": session_id})
        t2 = time.perf_counter()
        predictor.handle_request(
            {"type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": prompt}
        )
        per_frame = {}
        for resp in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session_id}):
            per_frame[resp["frame_index"]] = resp["outputs"]
        dt = time.perf_counter() - t2
        ids = sorted({int(i) for o in per_frame.values() for i in o["out_obj_ids"]})
        frames_with = sum(len(o["out_obj_ids"]) > 0 for o in per_frame.values())
        rows.append(
            {
                "prompt": prompt,
                "seconds": round(dt, 2),
                "frames": len(per_frame),
                "frames_with_object": frames_with,
                "track_ids": ids,
            }
        )
        log.info(
            "%-14s %.1fs for %d frames, %d with object, track ids %s",
            prompt,
            dt,
            len(per_frame),
            frames_with,
            ids,
        )

        # save an overlay for a few frames to eyeball tracking consistency
        for fi in np.linspace(0, len(frames) - 1, 4).round().astype(int):
            o = per_frame.get(int(fi))
            if o is None or len(o["out_obj_ids"]) == 0:
                continue
            img = np.array(Image.open(frames[fi]).convert("RGB")).astype(np.float32)
            for oid, m in zip(o["out_obj_ids"], o["out_binary_masks"], strict=True):
                color = np.array([(oid * 97) % 255, (oid * 57 + 120) % 255, (oid * 31 + 60) % 255])
                img[m] = 0.45 * img[m] + 0.55 * color
            Image.fromarray(img.astype(np.uint8)).save(out_dir / f"{prompt.replace(' ', '_')}_{fi:04d}.jpg")

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    peak = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "load_s": round(load_s, 1),
        "start_session_s": round(start_s, 2),
        "peak_vram_gb": round(peak, 2),
        "runs": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("peak VRAM %.2f GB", peak)
    predictor.shutdown()


if __name__ == "__main__":
    main()
