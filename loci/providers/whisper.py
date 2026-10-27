"""Local faster-whisper Transcriber (Phase 1 default, offline fallback after C1)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from loci.config import repo_path
from loci.providers.base import Utterance

log = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    def __init__(self, size: str = "small", device: str = "cuda") -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            size,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            download_root=str(repo_path("checkpoints/whisper")),
        )

    def transcribe(self, audio: Path) -> list[Utterance]:
        # audio.wav is 16 kHz mono from extract; pass samples (PyAV 19 breaks the built-in decoder)
        sr, pcm = wavfile.read(audio)
        if sr != 16000:
            raise ValueError(f"expected 16 kHz audio, got {sr}")
        if pcm.ndim > 1:
            pcm = pcm.mean(1)
        samples = pcm.astype(np.float32) / 32768.0
        segments, info = self.model.transcribe(samples, beam_size=5, vad_filter=True)
        out = [Utterance(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segments if s.text.strip()]
        log.info("transcribed %.1fs (%s): %d utterances", info.duration, info.language, len(out))
        return out
