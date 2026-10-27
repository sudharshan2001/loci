"""Local Glimmer implementations of Namer and VisionChecker."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from loci.llm import chat_json, image_part
from loci.providers.base import VisionAnswer

log = logging.getLogger(__name__)

HOME_AREAS = ["kitchen", "bedroom", "living room", "bathroom", "office", "hallway", "dining room", "balcony"]

NAME_PROMPT = (
    "These are frames from one walk through a {scene}. List the distinct physical objects a person might "
    "later ask about (furniture, appliances, devices, plants, containers, small personal items, signs). "
    'Use short lowercase noun phrases a segmentation model understands, e.g. "potted plant", '
    '"gas stove", "keys". No people, body parts, walls, floors or ceilings. Return only a JSON array '
    "of 15 to 40 strings."
)

AREA_PROMPT = 'Which part of a home is this frame from? Answer with JSON {{"area": one of {areas}}}.'


class GlimmerNamer:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def name_objects(self, images: list[Path], scene: str) -> list[str]:
        content: list[dict[str, Any]] = [image_part(p) for p in images]
        content.append({"type": "text", "text": NAME_PROMPT.format(scene="home" if scene == "home" else "street")})
        out = chat_json(self.base_url, content, strength="medium")
        if not isinstance(out, list):
            raise ValueError(f"expected a JSON array, got {type(out).__name__}")
        return [str(x) for x in out]

    def label_area(self, image: Path, scene: str) -> str:
        if scene == "street":
            return "street"
        out = chat_json(
            self.base_url,
            [image_part(image), {"type": "text", "text": AREA_PROMPT.format(areas=", ".join(HOME_AREAS))}],
            strength="low",
        )
        area = str(out.get("area", "")).lower().strip() if isinstance(out, dict) else ""
        return area if area in HOME_AREAS else "room"


class GlimmerVisionChecker:
    """Glimmer at low strength; if unsure, again at high strength with a tighter and a wider crop."""

    def __init__(self, base_url: str, min_confidence: float = 0.7) -> None:
        self.base_url = base_url
        self.min_confidence = min_confidence

    def _ask(self, images: list[Path], question: str, strength: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [image_part(p) for p in images]
        content.append(
            {
                "type": "text",
                "text": question + ' Include a "confidence" field from 0 to 1. Return only JSON.',
            }
        )
        out = chat_json(self.base_url, content, strength=strength)
        return out if isinstance(out, dict) else {"answer": out, "confidence": 0.0}

    def check(self, image: Path, question: str, tight: Path | None = None, wide: Path | None = None) -> VisionAnswer:
        ans = self._ask([image], question, "low")
        attempts = ["glimmer:low"]
        if float(ans.get("confidence", 0)) < self.min_confidence:
            crops = [p for p in (tight, image, wide) if p is not None]
            ans = self._ask(crops, question + " The images show the same object at different crops.", "high")
            attempts.append("glimmer:high")
        conf = float(ans.pop("confidence", 0))
        return VisionAnswer(answer=ans, confidence=conf, used_cloud=False, attempts=attempts)
