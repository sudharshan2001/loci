# Progress

## M0 – Repo and environment (Oct 27)

Works:
- `uv` project on Python 3.12, torch 2.11+cu128, CUDA available on the RTX 4080 SUPER (16 GB).
- SAM 3 and MapAnything installed as editable path deps from `third_party/`; faster-whisper installed.
- llama.cpp built with CUDA (sm_89); `llama-server` supports `--mmproj`, `--jinja`, `--reasoning-effort`.
- `pytest`: 7 passed. `ruff check`: clean.
- Weights: SAM 3.1 and benchmark-only models linked from existing local copies; MapAnything (Apache)
  downloaded; Glimmer mmproj downloaded, Q3/Q4 GGUFs downloading.

Measured (early SAM 3.1 image smoke test, 3 test frames, 5 prompts):
- load 20.9 s, encode ~0.12 s/frame, ~0.05 s per text prompt, peak VRAM 5.45 GB.

Known issues:
- sam3 pins `numpy<2` but MapAnything's rerun-sdk needs `numpy>=2`; overridden to `numpy>=2`.
- sam3 imports `pycocotools` and `psutil` without declaring them; added to our deps.
- Loading `sam3.1_multiplex.pt` into the image model reports 4 missing `convs.3` keys; masks look correct.
- SAM 3.1 multiplex video predictor defaults to FlashAttention-3 (Hopper only); must pass `use_fa3=False`.
- Hugging Face downloads stall and do not resume with hf_xet; using resumable parallel HTTP ranges instead.

## M1 – Capture / extract (Oct 27, partial)

Works:
- `loci extract <video>`: candidates at 4 fps, sharpest per equal time bin above the blur floor, cap 80;
  16 kHz mono `audio.wav`; raw frames only in `raw_tmp/`; `keyframes.json` with frame, time, sharpness.
- Tested on 11 short sample clips (3–37 s): 1–6 s per clip, max keyframe gap 0.5–1.25 s.

Pending:
- Acceptance on the developer's own `home.mp4` / `street.mp4`: lit burner and every shop sign must appear
  in at least one keyframe (add frames around them if not).

## M2 – Smoke tests and benchmarks (Oct 27, in progress)

MapAnything (Apache), 518 px long side, bf16, memory-efficient inference (sample clip, 86 keyframes):

| views | seconds | peak VRAM |
| --- | --- | --- |
| 20 | 3.3 | 10.98 GB |
| 40 | 6.6 | 11.55 GB |
| 60 | 10.5 | 12.13 GB |
| 80 | 14.7 | 13.03 GB |

- Model load 11.6 s warm (224 s on first run while backbone weights were fetched). Weights 4.96 GB.
- Median depth and metric scale stable across view counts (1.8 m, ×1.39). Desk sample reconstructs
  recognisably (desk, laptop, notebook, keyboard, mouse pad, wall corner); camera path plausible.
- 80 keyframes per session fits in 16 GB with ~3 GB headroom.

Glimmer Q3 (see `docs/benchmarks.md`):
- BF16 mmproj does not fit on the GPU next to Q3 (OOM). Q8_0 mmproj does: all layers + vision on GPU,
  14.7 GB, 49 tok/s, text 4–10 s, image 6–7 s, tool call 1.9 s with a correct `search_objects` call.
- Auto-fit (some layers on CPU) drops to 16 tok/s; vision on CPU makes images 27 s.
- Reasoning strength via system prompt works: high gives 2–3× the reasoning tokens of low.
- Decision: Q3_K_XL + Q8_0 mmproj fully on GPU. Q4 still to measure (expected slower, does not fit).
