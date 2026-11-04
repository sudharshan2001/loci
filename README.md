# Loci

**Loci gives AI glasses and robots a private memory of the places you move through.**

Named after the *method of loci*, the memory-palace technique of tying memories to places. Walk through your home
or down a street with a camera at eye level. Loci blurs faces, number plates and screens before anything is saved,
finds the objects, reads shop signs, notes the state of things like a lit stove or a fan left on, hears what you
say, and places everything in 3D with metric scale. Then ask:

- "Did I leave the stove on?" → *"When you left the kitchen at 0:15, the right burner was lit."* + the frame it saw
- "Where was <shop name>?" → *"On your left, about 280 m into the walk, at 4:41."* + the signboard frame
- "Where's my passport?" → *"I haven't seen a passport in any capture."*

Every answer cites the observations it used, is checked by a verifier against the memory before you see it, says
*when* a state was seen (never what it is now), and comes with a position in metres and a robot goal.

## How it works

```
video ─▶ extract ─▶ redact ─▶ name ─▶ track ─▶ geometry ─▶ lift ─▶ associate ─▶ relations ─▶ attributes
         keyframes  SAM 3.1   Glimmer SAM 3.1  MapAnything  2D→3D  objects     on/near/      signs, states
                    blur                                                       side of road  (Glimmer)
      ─▶ transcribe ─▶ store (SQLite) ─▶ leaving check (agent) ─▶ ask (agent + verifier) ─▶ viewer / API
         Muse Voice                       alerts                   Muse Spark or Glimmer
```

`loci build <video> --scene home|street` runs every stage, each in its own subprocess so only one large model is on
the GPU at a time. Every model output is cached on disk.

| Role | Model | Where |
| --- | --- | --- |
| Redaction, segmentation, tracking | SAM 3.1 | your GPU |
| Camera poses, metric depth | MapAnything (Apache) | your GPU |
| Object names, rooms, sign reading, states, leaving check, offline answers | **Muse Glimmer 30B** (GGUF, llama.cpp) | your GPU |
| Answers when online | Muse Spark 1.3 | Meta Model API |
| Speech (25 languages, mixed in one walk) | Muse Voice Transcribe 1.0 | Meta Model API |
| Speech offline, and all street audio | faster-whisper | your machine |

Cloud is optional: `LOCI_CLOUD=0` (or `cloud.enabled: false`) runs everything on device. When cloud is on and a
call fails, Glimmer answers instead. The viewer shows a badge for which model answered.

## Runs on open weights

The memory is built by **Muse Glimmer 30B**, Meta's open-weight model (Apache 2.0), on one consumer GPU:

- Model: `Muse-Glimmer-30B-UD-Q3_K_XL.gguf` (13.4 GB) with `mmproj-Muse-Glimmer-30B-Q8_0.gguf` (2.1 GB)
- GPU: NVIDIA RTX 4080 SUPER, 16 GB (14.8 GB peak), CUDA 12.8
- Server: llama.cpp `llama-server` (build e86f5fd)

```bash
llama-server -m checkpoints/glimmer/Muse-Glimmer-30B-UD-Q3_K_XL.gguf \
  --mmproj checkpoints/glimmer/mmproj-Muse-Glimmer-30B-Q8_0.gguf --jinja -c 16384 -ngl 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8001
```

Glimmer names every object and room, reads every shop sign (as written and in English), judges every state (stove,
fan, lights, laundry), describes objects for search, and runs the leaving-check agent that decides what to flag.
With cloud off it also answers every question with the same tools and verifier. Check it on your machine:

```bash
uv run python scripts/verify_open_weights.py "Did I leave the fan on?"
```

prints the GGUF the server has loaded, its hash, the GPU, and one verified answer.

## Privacy

- Faces, number plates, computer and phone screens and documents are blurred before any frame is written; raw
  frames are deleted after redaction. Counts go to `privacy_ledger.json` (shown in the viewer).
- On the street nothing about people is stored beyond those counts; shop signs stay readable.
- Cloud calls get questions, memory facts and blurred crops only. Street audio never leaves the machine.
- `loci forget <object> --yes` blurs an object out of the stored frames, cuts its 3D points and deletes its
  evidence.

## Quick start

```bash
uv sync
uv run loci build data/raw/home.mp4 --scene home        # # builds the memory; about 30 min for a 1-minute walk on one 16 GB GPU
uv run loci ask "Did I leave the stove on?"
uv run loci serve                                        # viewer at http://127.0.0.1:8000
```

Needs Python 3.12, an NVIDIA GPU with 16 GB, `third_party/sam3`, `third_party/map-anything`, llama.cpp built with
CUDA, and the checkpoints listed in `configs/default.yaml`. For cloud answers and speech put `MODEL_API_KEY=...`
in `.env`.

## Limits

- Built after the walk, not live (answers take seconds).
- Fast turns break the 3D: walk slowly and turn over 2–3 seconds (`loci coach` says which parts to re-walk).
- Rooms are named from single frames and are sometimes left unnamed rather than guessed.

## Licence

Apache 2.0 (see `LICENSE`, and `NOTICE` for the models).
