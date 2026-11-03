"""Muse Voice Transcribe (Meta Model API) Transcriber: handles code-switching, e.g. English and Tamil in one walk."""

from __future__ import annotations

import logging
from pathlib import Path

from loci.cloud import meta_api
from loci.providers.base import Utterance

log = logging.getLogger(__name__)


class MuseVoiceTranscriber:
    def __init__(self, language_bias: list[str] | None = None, keywords: list[str] | None = None) -> None:
        self.language_bias = language_bias
        self.keywords = keywords

    def transcribe(self, audio: Path) -> list[Utterance]:
        res = meta_api.transcribe(audio, self.language_bias, self.keywords)
        out = [
            Utterance(round(t["startMs"] / 1000, 2), round(t["endMs"] / 1000, 2), t["transcript"].strip())
            for t in res.get("turns") or []
            if (t.get("transcript") or "").strip()
        ]
        log.info("transcribed %.1fs (muse voice): %d utterances", (res.get("audioDurationMs") or 0) / 1000, len(out))
        return out
