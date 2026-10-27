# Loci – build guide

**Loci gives AI glasses and robots a private memory of the places you move through.**
Named after the method of loci (the memory-palace technique). Hackathon project, solo developer (strong in
Python, CV and 3D). The deadline is fixed: working software beats perfect software.

## 1. How to work

- Work milestone by milestone, in order. Finish a milestone's acceptance check before starting the next.
- After each milestone: run its check, commit (`M3: privacy redaction with SAM 3.1`), and append to
  `PROGRESS.md` what works, measured numbers (VRAM, seconds, accuracy) and known issues.
- **Never guess a library's API.** For every model, first read the installed package's README, examples and
  source, then write a short smoke test that runs, then integrate. If docs and memory disagree, docs win.
- Run each GPU-heavy stage as its own subprocess (one CLI command per stage) so VRAM is freed between
  stages. Never load two large models at once.
- Cache every model output on disk, keyed by a hash of its inputs and config.
- Stop and ask the human only at the HUMAN CHECKPOINTS, or when a decision cannot be undone.
- Keep the repo clean: typed Python, small modules, config in `configs/default.yaml`, `logging` not prints,
  `pytest` for pure functions, `ruff` clean.

### Build order: local first, cloud second

- **Phase 1 – local only (M0–M9).** No Meta Model API calls, `cloud.enabled: false`. Local stand-ins:
  hard visual checks ask Glimmer again at higher reasoning strength with a tighter and a wider crop;
  speech uses faster-whisper.
- **HUMAN CHECKPOINT B (end of M9).** Show the local pipeline working: demo questions, answers, evidence
  frames, numbers from `PROGRESS.md`. Wait for a go-ahead before touching the cloud.
- **Phase 2 – Meta cloud models (C1, then M10–M13).** Muse Spark 1.3 and Muse Voice Transcribe behind the
  same interfaces. Local stays the default and the fallback.

Every model role sits behind a small interface in `loci/providers/` (`Namer`, `VisionChecker`,
`Transcriber`). Swapping is one line of config, never a code change in the pipeline or agent.

## 2. Hackathon constraints

- Meta Global AI Developer Hackathon, Indie Developer track, solo.
- Build period Oct 27 – Nov 4, 2026. **Submission deadline: Nov 4, 9:00 PM ET = Nov 5, 7:30 AM IST.**
- Repo `github.com/sudharshan2001/loci`: private during the build, **must be made public before
  submission**. Only work from Oct 27 counts; commit history must show steady progress.
- The 3-minute demo video is the main judging artifact. Finalists demo live, so question answering must
  work reliably on a pre-built memory.
- Judging: Impact 30%, Use of Meta AI tooling 25%, Technical depth 20%, Demo 25%.
- Banned categories include medical advice, basic RAG and simple image analysers. **Never describe Loci as a
  health, medical, cognitive or memory-loss tool** anywhere. It is a spatial memory for everyday objects.
- Footage is the developer's own home plus a walk along a public street. Every face and number plate (and,
  at home, screens and documents) is blurred before anything is saved. Shop signboards stay readable.
- Only the developer's own footage goes in the repo, eval, README or video.

### Bonus prizes we target

| Prize | Plan |
| --- | --- |
| Best Build on Meta Open Weights ($10K) | Glimmer does the core work locally (agent, naming, sign reading, stove check). README section "Runs on open weights": exact GGUF file, llama.cpp command, GPU. `scripts/verify_open_weights.py` prints the running model, GPU, and answers one demo question. Video: 5 s shot of llama-server + nvidia-smi. Glimmer endpoint is a config value (`models.glimmer.base_url`), so a hosted Glimmer (Baseten/Modal, OpenAI-compatible) is a config switch if partner inference turns out to be required. |
| Best Agentic Use-Case ($5K) | **Leaving check**: runs automatically after every `build`. The agent decides what is worth flagging and writes alerts with evidence, e.g. "Burner was lit when you left the kitchen at 0:31", "Keys left on the desk", "You said you'd water the plant tomorrow". Same tools and verifier as `ask`; state alerts are always "as of <time>". |
| Most Viral ($20K) | Separate 30–45 s vertical clip posted on X before Nov 8, 11:59 PM ET. Hook in first 2 s, 3D fly-to, blurred faces/plates, stove alert, captions. Viewer must read well in a 9:16 crop. |
| Community Vote ($10K) | Strong gallery thumbnail and title. No code. |
| Best Creative Build | Skipped (needs Muse Image/Wonder). |

## 3. The product

Walk with a phone at eye level (standing in for glasses) through your home or down a street. Loci finds the
objects, reads shop signs, notes the state of things like a lit stove, places everything in 3D with metric
scale, and stores it locally. Then ask "did I leave the stove on?" or "where was Sudarshan Silks?" and get:

- a spoken-style answer: "When you left the kitchen at 0:31, the left burner was lit with nothing on it",
- the object highlighted in a 3D point cloud, with a path from where you stand,
- the redacted video frame as proof, with timestamp and confidence,
- a robot-ready goal (object position in metres, in that place's frame) as JSON.

The agent is verifiable: every answer cites the observations it used, a verifier checks them, and when the
evidence is weak it says it is not sure and reports where the object was last seen instead of guessing.
State answers always say when the state was seen ("as of 0:31"), never claim to know the state now.

Demo questions:

**Home walkthrough** (kitchen with one burner lit and nothing on it, then the room with PC, plants, desk,
plus small items placed on purpose):
1. "Did I leave the stove on?" → burner state, timestamp, proof frame (Glimmer; from C1 escalates to Muse Spark).
2. "What's on my desk?" → each object with its position.
3. "Where are my plants?" / "Which plant is closest to my PC?" → location text and distance in metres.
4. "Where did I leave my keys?" → a small placed item.
5. "What did I say about the plant?" → spoken note linked to the object in view.

**Street walk** (passing shops including Sudarshan Silks):
6. "Where was Sudarshan Silks?" → "On your right, about 80 m into the walk, at 1:12" + signboard frame.
   The name comes from reading the sign, not a map.
7. "What shops did I pass?" → names in order, distance along the walk, side of road.

**Refusal:** "Where's my passport?" (never filmed) → "I haven't seen a passport in any capture."
**Optional (second home capture):** 8. "What moved since this morning?"

## 4. Hardware and environment

- RTX 4080 SUPER 16 GB, 32 GB RAM, Ubuntu (native Linux), CUDA 12.8, driver 580.
- Python 3.12 via `uv` (`uv sync`, `uv run ...`). torch 2.11 cu128.
- `third_party/sam3` and `third_party/map-anything` are editable path deps (gitignored clones).
- numpy is overridden to `>=2` (sam3 pins `<2`; mapanything's rerun-sdk needs `>=2`). SAM 3 works with it.
- llama.cpp built with CUDA at `../llama.cpp/build/bin/llama-server` (outside the repo).
- `checkpoints/` (gitignored): `sam3.1/`, `map-anything-apache/`, `glimmer/` (UD-Q3_K_XL, UD-Q4_K_XL,
  mmproj Q8_0 + BF16). Also linked but unused: `DA3NESTED-GIANT-LARGE-1.1`, `moge-3-vit{l,g}`,
  `vggt_omega_1b_512.pt`.
- Measure free VRAM before each stage and log the peak after it.

## 5. Models

| Role | Model | Runs | Notes |
| --- | --- | --- | --- |
| Segmentation, tracking, redaction | SAM 3.1 | Local GPU | `checkpoints/sam3.1/sam3.1_multiplex.pt`. Image model loads it (4 missing `convs.3` keys, results fine). Multiplex video predictor defaults `use_fa3=True` (Hopper only): pass `False`. |
| Poses, depth, metric scale | MapAnything (Apache) | Local GPU | `MapAnything.from_pretrained(...)`, `load_images`, `model.infer(views, memory_efficient_inference=True, minibatch_size=...)`. |
| Agent, local vision, naming | Muse Glimmer 30B GGUF | llama.cpp | Apache 2.0. Sampling temp 1.0, top_p 0.95, top_k 64. **Reasoning strength is set in the system prompt: `Reasoning strength: low|medium|high|xhigh`.** Benchmark Q3 vs Q4. |
| Speech (Phase 1) | faster-whisper | Local subprocess | MIT. Offline fallback after C1. |
| Cloud second opinion (Phase 2) | Muse Spark 1.3 | Meta Model API | Redacted crops only. Everything works with `--no-cloud`. |
| Speech (Phase 2) | Muse Voice Transcribe 1.0 | Meta Model API | Default transcriber from C1. |
| Not used | VGGT-Omega, Depth Anything 3, MoGe-3, YOLO | – | **Developer decision (Oct 27): MapAnything is the only geometry model.** No comparison runs. YOLO excluded (AGPL). |

Meta Model API docs: https://dev.meta.ai/docs/overview, /quickstart, /models, /pricing-rate-limits.
All Meta API calls go through `loci/cloud/meta_api.py` (retries, timeouts, disk cache, key never logged).

llama-server start (confirm flags with `--help`):
```
llama-server -m checkpoints/glimmer/Muse-Glimmer-30B-UD-Q3_K_XL.gguf \
  --mmproj checkpoints/glimmer/mmproj-Muse-Glimmer-30B-Q8_0.gguf --jinja -c 16384 -ngl 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8001
```
Start it only for stages that need it; stop it before SAM 3.1 or MapAnything run.

## 6. Pipeline

Each session has a scene type `home` or `street` (`--scene`, default `home`), which picks redaction prompts,
default vocabulary and how locations are described. Home and street sessions are never registered together.

1. `extract` – sample candidate frames, drop blurry ones (variance of Laplacian), even coverage, cap at N
   keyframes (start 80). Audio to `audio.wav`. Raw frames to `raw_tmp/` only.
2. `redact` – SAM 3.1 image mode with the scene's prompts. Home: face, computer screen, phone screen,
   document with text. Street: face, license plate (no document prompts, signs stay readable). Gaussian-blur
   masks, write redacted keyframes, delete `raw_tmp/` (unless `--keep-raw`). Counts to `privacy_ledger.json`.
   **Every later stage uses redacted frames only.**
3. `name` – Glimmer on 6–8 spread keyframes → 15–40 noun phrases (JSON). Merge with scene vocab, normalise,
   dedupe. Area label for every 5th keyframe, fill the rest by nearest timestamp.
4. `track` – SAM 3.1 on the keyframe sequence with the phrases → per-keyframe instances (label, track id,
   score, bbox, RLE mask).
5. `geometry` – MapAnything: poses, intrinsics, metric depth, confidence → `geometry.npz`, `scene.ply`
   (≤ 500k points). Street: drop person/vehicle masks and far low-confidence points.
6. `lift` – resize masks to depth resolution (track each model's resize/crop/pad exactly), back-project with
   confidence filtering, remove outliers, per-observation centroid and 3D box.
7. `associate` – same track id, then same label + 3D overlap or centroid distance. Object centroid, extent,
   first/last seen, best observation.
8. `relations` – home: on/inside/near/left_of/right_of relative to anchors, distance to door and start.
   Street: distance_along_walk_m, side, time passed.
9. `attributes` – Glimmer on best crops: sign text (as written + English; English and Kannada), structured
   state for stateful objects with confidence and timestamp; low confidence → Glimmer at higher strength
   (Phase 1) → Muse Spark (C1, if cloud allowed).
10. `transcribe` – configured Transcriber; link utterances to objects visible within ±2 s.
11. `store` – SQLite.
12. `leaving_check` – agent writes proactive alerts with evidence (see bonus prizes).

`loci build <video> --scene home|street` runs it all.

Session folder: `data/sessions/<id>/` with `video.mp4 audio.wav keyframes/ keyframes.json
privacy_ledger.json vocab.json tracks.npz tracks.json geometry.npz scene.ply objects/<id>.npz`;
memory in `data/memory.sqlite` (tables: sessions, keyframes, observations, objects, relations, utterances,
utterance_links, alerts).

## 7. Agent

Glimmer with tools via llama-server (OpenAI-compatible `/v1/chat/completions`, `base_url` from config).
Tools: `search_objects`, `get_state`, `places_passed`, `get_object`, `objects_near`, `objects_on`, `history`,
`what_changed`, `search_notes`, `get_evidence`, `look`.

Answer contract:
```json
{"answer": "...", "object_id": "...", "position_m": [x, y, z], "location_text": "...",
 "seen_at": "0:31", "evidence": ["obs_..."], "confidence": 0.0, "used_cloud": false, "robot_goal": {...}}
```

Verifier (plain code, after every answer): cited observations exist, belong to the object, labels match;
position inside the object's box; relation words match stored relations; state claims match and include
when seen; shop names fuzzy-match sign text; street distance and side match. On failure one retry with the
error, then a "not sure" answer with the last confirmed sighting. Log every step for the viewer.

## 8. API and viewer

FastAPI: `POST /ask`, `POST /ask_audio`, `GET /sessions`, `GET /scene/{session}`, `GET /objects`,
`GET /evidence/{observation_id}.jpg`, `GET /privacy/{session}`, `GET /alerts/{session}`.
Viewer: one static three.js page. Load `scene.ply`, highlight the answered object, fly to it, draw a path
from "you are here". Side panel: answer, evidence frame, timestamp, confidence, local/cloud badge, robot goal
JSON, privacy ledger, alerts. Session picker; street sessions show the walk path with shop markers.
Dark, readable at 1080p and in a 9:16 crop, no debug clutter.

## 9. Milestones and schedule

| Day | Milestones |
| --- | --- |
| Mon Oct 27 | M0 repo/env, M1 extract, downloads |
| Tue Oct 28 | M2 smoke tests + `docs/benchmarks.md`, M3 redaction, M4 naming |
| Wed Oct 29 | M5 tracking, M6 geometry (MapAnything only), start M7 |
| Thu Oct 30 | M7 lift/associate/relations/attributes, M8 `loci build`, M9 agent + verifier → **Checkpoint B** |
| Fri Oct 31 | C1 Meta cloud models, M10 (if evening capture exists) |
| Sat Nov 1 | M11 speech, leaving check, verify_open_weights, M12 API + viewer |
| Sun Nov 2 | M12 polish, M13 eval |
| Mon Nov 3 | README, docs, demo script; developer records main video + vertical clip |
| Tue Nov 4 | Buffer, make repo public, submit |

Acceptance checks:
- **M0** pytest passes, CUDA available, nvidia-smi shows the 4080 Super.
- **M1** keyframes sharp, evenly spread, count/timing reported; lit burner and every shop sign in ≥ 1 keyframe.
- **M2** smoke script per model in `scripts/smoke/`; `docs/benchmarks.md` with peak VRAM, seconds, tok/s;
  pick keyframe budget and Glimmer quant.
- **M3** no visible faces, plates or screen content in any saved keyframe; signs readable; ledger right;
  blur unit test.
- **M4** vocab covers obvious objects; area labels split kitchen/room; JSON always parses (retry).
- **M5** overlay video of masks with labels in `debug/`.
- **M6** `scene.ply` looks like the place; plausible camera path; one tape-measured distance within ~10%.
- **M7** back-projection unit tests; each object once at a sensible place; 5 hand-checked relations; stove
  state right; every readable shop name read correctly, in walk order.
- **M8** `loci build` end to end on both videos, prints an object table.
- **M9** `loci ask` with `--no-cloud`: demo questions 1–7 with evidence, refusal says not sure; ~15-question
  local eval with accuracy and median latency in `PROGRESS.md`.
- **C1** Meta API client + smoke tests; cloud providers; local vs cloud table. If the API is unreachable,
  record the error, keep cloud off, continue.
- **M10** two home sessions in one frame; `what_changed`; question 8 works.
- **M11** question 5 works with both transcribers; `/ask_audio` works.
- **M12** every demo question in the browser with highlight, path, evidence, badges; works without API key.
- **M13** `eval/questions.yaml` (~25, home + street, ≥ 3 refusals), `scripts/eval.py` → `eval/results.md`;
  full README; `docs/demo_script.md`.

## 10. Non-goals

Gaussian splatting, real-time streaming, mobile app, training/fine-tuning, user accounts, database server,
Muse Image.

## 11. Guardrails

- Redact before persist. Cloud calls receive redacted crops only. `--no-cloud` works everywhere.
- Phase 1 makes no Meta Model API calls.
- Speech is credited to Muse Voice Transcribe once C1 is done; faster-whisper listed in NOTICE as fallback.
- Street: store nothing about people beyond ledger counts.
- State answers and alerts are always "as of when seen".
- Never log or print API keys. Never commit `data/`, `checkpoints/`, `clips/`, `.env`.
- No health, medical or cognitive claims anywhere.
- Keep `NOTICE` current; MapAnything (Apache) in the shipped pipeline, noncommercial models benchmark only.
- If blocked, switch to the fallback, note it in `PROGRESS.md`, continue with the next unblocked milestone.
