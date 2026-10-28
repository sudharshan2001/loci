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
        # whisper invents text from noise (street sounds became a Vietnamese "subscribe" line): keep only
        # segments the model itself rates as speech and decodes with reasonable confidence
        out = [
            Utterance(round(seg.start, 2), round(seg.end, 2), seg.text.strip())
            for seg in segments
            if seg.text.strip() and seg.no_speech_prob < 0.5 and seg.avg_logprob > -1.0 and seg.compression_ratio < 2.4
        ]
        log.info("transcribed %.1fs (%s): %d utterances", info.duration, info.language, len(out))
        return out
