"""Model-role interfaces. Local implementations now, cloud ones from C1; chosen by config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class VisionAnswer:
    answer: dict[str, Any]
    confidence: float
    used_cloud: bool = False
    attempts: list[str] = field(default_factory=list)


@dataclass
class Utterance:
    start_s: float
    end_s: float
    text: str
    speaker: str | None = None


class Namer(Protocol):
    def name_objects(self, images: list[Path], scene: str) -> list[str]: ...

    def label_area(self, image: Path, scene: str) -> str: ...


class VisionChecker(Protocol):
    def check(self, image: Path, question: str) -> VisionAnswer: ...


class Transcriber(Protocol):
    def transcribe(self, audio: Path) -> list[Utterance]: ...
