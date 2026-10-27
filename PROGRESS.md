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
