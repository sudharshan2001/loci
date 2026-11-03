# Demo script

Main video: 3 minutes, 16:9, recorded on screen with voice-over. Vertical clip: 30–45 s, 9:16, captions on.
Everything shown is the developer's own footage, already blurred by Loci. Keep frames 0007–0008 of `home3`
(the item held in hand) out of the video.

Before recording: run `loci warm eval/questions.yaml` with `demo.cache_answers: true`, so every answer is instant
and the same on every take. Start `loci serve` and open the viewer at http://127.0.0.1:8000.

## Main video (3:00)

| Time | Screen | Voice-over |
| --- | --- | --- |
| 0:00–0:10 | Phone at eye level walking into the kitchen (raw clip, faces blurred) | "Glasses and robots see everything you do. Loci gives them a memory of it, and keeps it on your machine." |
| 0:10–0:30 | `loci build` running: stages ticking, blurred faces and screens in the privacy panel | "One walk through the home, one down a street. Loci blurs faces, plates and screens before anything is saved, finds every object with SAM 3.1, and rebuilds the place in 3D with MapAnything." |
| 0:30–0:55 | Viewer, home: ask **"Did I leave the stove on?"** → answer, fly-to, evidence frame, "as of 0:15" | "It answers with proof: the frame it saw, when it saw it, and never a claim about right now." |
| 0:55–1:15 | Leaving check alerts for `home3`: fan on, hallway light on, three spoken notes | "When you leave, the agent decides what's worth a heads-up on its own: the fan you left on, the light, and what you said you'd do." |
| 1:15–1:30 | Ask **"What do I have to do tomorrow?"**, then the Tamil **"குப்பை பற்றி நான் என்ன சொன்னேன்?"** | "It heard you in English and in Tamil, with Muse Voice Transcribe." |
| 1:30–1:50 | Ask **"What's on my desk?"** and **"How far is the desk from the door?"** (`home`), robot goal JSON | "Every object has a position in metres, so a robot gets a goal and a path, not just a sentence." |
| 1:50–2:20 | Street: **"Where was Lenovo?"** → "on your left, about 278 m in" + signboard frame; **"What shops did I pass?"** | "On the street it reads the signs, in English and Kannada. The Lenovo shop is 260 m from my office; Loci says 278." |
| 2:20–2:35 | **"Where's my passport?"** → "I haven't seen a passport in any capture." | "When it hasn't seen something, it says so instead of guessing. Every answer is checked against the memory before you see it." |
| 2:35–2:50 | Badge flips: cloud answer (Muse Spark, ~7 s) → unplug Wi-Fi, same question → Glimmer on device | "Muse Spark answers fast in the cloud. Pull the plug and Muse Glimmer answers on this GPU. Open weights, nothing leaves the house." |
| 2:50–3:00 | Title card: Loci, repo URL, "Muse Glimmer · Muse Spark · Muse Voice · SAM 3.1 · MapAnything" | "Loci. A private memory of the places you move through." |

Shots to grab separately:
- 5 s of `llama-server` loading `Muse-Glimmer-30B-UD-Q3_K_XL.gguf` next to `nvidia-smi` (open-weights prize).
- `uv run python scripts/verify_open_weights.py "Did I leave the fan on?"` output.

## Vertical clip (30–45 s, 9:16)

| Time | Shot | Caption |
| --- | --- | --- |
| 0:00–0:02 | Close-up: the stove flame | "Did I leave the stove on?" |
| 0:02–0:08 | Viewer, 9:16 crop: answer card + fly-to in the 3D kitchen | "Lit, as of 0:15. Here's the frame." |
| 0:08–0:15 | Walk footage with faces and plates blurring in | "It blurs every face and plate before saving anything." |
| 0:15–0:25 | Street: Lenovo sign highlighted on the walk path | "Where was Lenovo? 278 m in, on your left." |
| 0:25–0:35 | Alerts sliding in: fan on, trash (Tamil) | "And it reminds you what you said you'd do." |
| 0:35–0:40 | Logo + "runs on open weights" | "Loci · Muse Glimmer on your own GPU" |

Post on X before Nov 8, 11:59 PM ET.

## Questions to avoid on camera

- "Which plant is closest to my PC?": `home3`'s 3D is unreliable (fast turns), so its distances are not trusted.
- "Where are my keys?": no keys in any capture (it refuses correctly, but it is not the demo).
- "Where was Kalyan Jewellers?": two boards; the 46 m one is doubtful.
