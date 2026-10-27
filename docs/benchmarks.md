# Benchmarks

Hardware: RTX 4080 SUPER (16 GB), 32 GB RAM, Ubuntu, CUDA 12.8. All numbers measured with the scripts in
`scripts/smoke/`. Peak VRAM is `torch.cuda.max_memory_allocated` for PyTorch models and the `nvidia-smi`
delta for llama-server.

## Muse Glimmer 30B (llama.cpp, Q3)

`Muse-Glimmer-30B-UD-Q3_K_XL.gguf` (13.4 GB), ctx 16384, KV cache q8_0, sampling temp 1.0 / top_p 0.95 /
top_k 64. Reasoning strength set in the system prompt. Image test: one 848×478 desk frame.

| Config | GPU | Gen tok/s | Text low / high | Image low / high | Tool call |
| --- | --- | --- | --- | --- | --- |
| all layers GPU + mmproj **BF16 GPU** | – | – | out of memory | – | – |
| all layers GPU + mmproj BF16 **CPU** | 12.4 GB | 49 | 4.9 s / 9.1 s | 27.4 s / 29.3 s | 1.7 s ✓ |
| **auto-fit** layers + mmproj BF16 GPU | 15.0 GB | 16 | 12.1 s / 28.5 s | 14.0 s / 36.0 s | 5.4 s ✓ |
| **all layers GPU + mmproj Q8_0 GPU** (chosen) | 14.7 GB | **49** | **4.3 s / 9.5 s** | **6.1 s / 7.4 s** | **1.9 s ✓** |

- High reasoning strength produces ~2–3× the reasoning tokens of low; the speed per token is unchanged.
- Server start: 50–60 s with the file in page cache, 139–463 s cold from the data HDD (see `improvement.md`).
- Q4_K_XL (15.9 GB) cannot fit with any vision projector and the KV cache; expect auto-fit speeds (~16 tok/s).
  To be measured when the download finishes.

## MapAnything (Apache)

518 px long side, bf16, `memory_efficient_inference=True`. 86-keyframe sample clip.

| Views | Seconds | Peak VRAM |
| --- | --- | --- |
| 20 | 3.3 | 10.98 GB |
| 40 | 6.6 | 11.55 GB |
| 60 | 10.5 | 12.13 GB |
| 80 | 14.7 | 13.03 GB |

Model load 11.6 s warm. Median depth (1.8 m) and metric scale (×1.39) stable across view counts.
**Keyframe budget: 80** (fits with ~3 GB headroom).

## SAM 3.1 (image mode)

`sam3.1_multiplex.pt`, bf16 autocast, 1008 px. Load 20.9 s, image encode ~0.12 s, ~0.05 s per text prompt,
peak 5.45 GB. For 80 keyframes × 4 home redaction prompts: ~26 s estimated.

## Decisions

- Glimmer quant: **Q3_K_XL with the Q8_0 mmproj, fully on GPU.**
- Keyframe budget: **80.**

## SAM 3.1 (video, multiplex predictor)

Built via `loci.sam3_compat.build_video_predictor` (`use_fa3=False`, 4-frame grounding/postprocess
batches, `init_state` kwarg fix). 23 keyframes at 848×478, one text prompt per pass.

| Prompt | Seconds | Tracks |
| --- | --- | --- |
| keyboard | 5.5 | 4 |
| laptop | 4.2 | 2 |
| computer screen | 4.2 | 3 |
| face | 4.4 | 1 |

Load 12.9 s, peak VRAM 9.05 GB (the default 16-frame batches run out of memory on 16 GB).
~0.19 s per frame per prompt. On sparse keyframes some objects drop out mid-sequence (the external keyboard
is lost by the last frame), so M5 should not rely on video tracking alone.

## faster-whisper (small, float16, CUDA)

Load 5.1 s (cached). 27.6 s Hindi speech in 5.2 s (incl. warm-up), 5.4 s in 0.23 s; language detected
correctly (hi, 0.81–0.85). Hindi text quality from `small` is rough; English notes expected to be fine.
Audio is passed as samples read from our 16 kHz `audio.wav` (PyAV 19 breaks faster-whisper's decoder).
