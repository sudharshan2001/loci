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

SAM 3.1 video and faster-whisper (see `docs/benchmarks.md`):
- SAM 3.1 video works through `loci/sam3_compat.py`: needs `use_fa3=False`, 4-frame batches instead of the
  hard-coded 16 (OOM otherwise; 9.05 GB peak now), and a fix for `start_session` passing kwargs that the
  multiplex `init_state` does not accept. ~0.19 s per frame per prompt; one prompt per pass.
- Known issue: on sparse keyframes the video tracker drops some objects. Plan for M5: SAM 3.1 image mode per
  keyframe for detection, video tracks only as an association hint; identity resolved in 3D (M7).
- faster-whisper small works (5× real time or better); passes WAV samples directly because PyAV 19 breaks
  its built-in decoder.

## M3–M9 – Pipeline and agent, first end-to-end run (Oct 27–28, on a sample desk clip)

All stages are written and run end to end with `loci build`; real-footage acceptance checks wait for the
developer's own `home.mp4` / `street.mp4` (Oct 28).

Works (desk sample, 23 keyframes):
- redact: screens and on-screen faces blurred, keyboard/mouse/notebook stay sharp. 34 faces, 56 screens.
- name: Glimmer lists the right 16 objects (two laptops, external keyboard, mouse, mouse pad, pink notebook,
  pen, webcam, outlet, monitor, chair).
- track: 114 s for 23 frames x 27 phrases. A confused-part filter drops "keys" detections inside keyboards
  and laptops (all 196 false "keys" were keyboard keys).
- geometry: MapAnything + gravity alignment. "Up" now comes from the largest horizontal plane (RANSAC); the
  camera-based estimate was 16° off when looking down at a desk, which broke every "on" relation.
- lift/associate/relations: 21 objects; 10 correctly "on the desk".
- agent: `loci ask` answers with verified evidence, position and robot goal. 7 of 7 test questions verified
  on the first attempt after fixes; latency 14–49 s (refusals 16–23 s).

Fixes made after the first run:
- Object boxes are the robust union of all views (median boxes made the desk 0.5 m wide).
- Area labels smoothed (isolated disagreeing samples are overruled).
- Agent: empty search returns a "nothing matching" note; last step offers no tools so it must answer;
  server 500s (malformed tool call at temperature 1.0) are retried; one merged system prompt.

Known issues:
- The far end of a long desk is "left of the desk": no single view sees the whole desk.
- Duplicate objects for things seen only at the frame edge (three chairs for one).
- Screen on/off is unreliable after blurring (blurred screens look dark).
- Glimmer server start is 1–9 min from the data HDD (see improvement.md). geometry stage 5 min, mostly load.

## M10–M12 early work (Oct 28, ahead of schedule while waiting for footage)

- Association: second merge pass for overlapping same-label objects and a minimum score for single
  sightings (bedroom sample 49 -> 41 objects, desk 21 -> 18).
- Attributes are cached per observation, so re-running association no longer needs Glimmer.
- API (`loci serve`): /ask, /ask_audio (whisper on CPU so it never competes with Glimmer for VRAM),
  /sessions, /scene, /objects, /evidence (redacted frame with mask outline), /privacy, /alerts.
- Viewer: point cloud, camera path, "you are here", highlight + fly-to + dashed path on answer, evidence
  frame, local/verified/confidence badges, robot goal JSON, privacy ledger, voice questions, street shop
  markers. Checked with headless Chromium at 1920x1080: "Where's my mouse?" answered in 14 s, verified.
- Leaving check (`loci check`, last stage of `loci build`): Glimmer picks what to flag from states,
  valuables and spoken notes; every alert passes the verifier. Not yet run on footage with a stateful object.
- `scripts/eval.py`: accuracy, refusals, first-attempt verification, median latency -> eval/results.md.
