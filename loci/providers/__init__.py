"""Provider factory: the config picks the implementation for each model role."""

from __future__ import annotations

from loci.config import load_config
from loci.providers.base import Namer, Transcriber, VisionChecker


def get_namer(base_url: str) -> Namer:
    kind = load_config()["providers"]["namer"]
    if kind == "glimmer":
        from loci.providers.glimmer import GlimmerNamer

        return GlimmerNamer(base_url)
    raise ValueError(f"unknown namer {kind!r}")


def get_vision_checker(base_url: str) -> VisionChecker:
    kind = load_config()["providers"]["vision_checker"]
    if kind == "glimmer":
        from loci.providers.glimmer import GlimmerVisionChecker

        return GlimmerVisionChecker(base_url)
    raise ValueError(f"unknown vision checker {kind!r}")


def get_transcriber(kind: str | None = None) -> Transcriber:
    kind = kind or load_config()["providers"]["transcriber"]
    if kind == "faster_whisper":
        from loci.providers.whisper import FasterWhisperTranscriber

        return FasterWhisperTranscriber(load_config()["models"]["whisper"]["size"])
    if kind == "muse_voice":
        from loci.providers.muse_voice import MuseVoiceTranscriber

        voice = load_config()["models"].get("muse_voice", {})
        return MuseVoiceTranscriber(voice.get("language_bias"), voice.get("keywords"))
    raise ValueError(f"unknown transcriber {kind!r}")
