"""Paths and metadata for one capture session folder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loci.config import load_config, repo_path


@dataclass(frozen=True)
class Session:
    root: Path

    @classmethod
    def named(cls, session_id: str) -> Session:
        return cls(repo_path(load_config()["paths"]["sessions"]) / session_id)

    @property
    def id(self) -> str:
        return self.root.name

    # files and folders
    raw_dir = property(lambda self: self.root / "raw_tmp")
    keyframes_dir = property(lambda self: self.root / "keyframes")
    keyframes_json = property(lambda self: self.root / "keyframes.json")
    audio = property(lambda self: self.root / "audio.wav")
    ledger = property(lambda self: self.root / "privacy_ledger.json")
    vocab = property(lambda self: self.root / "vocab.json")
    tracks_json = property(lambda self: self.root / "tracks.json")
    geometry = property(lambda self: self.root / "geometry.npz")
    ply = property(lambda self: self.root / "scene.ply")
    observations = property(lambda self: self.root / "observations.json")
    objects = property(lambda self: self.root / "objects.json")
    attributes = property(lambda self: self.root / "attributes.json")
    utterances = property(lambda self: self.root / "utterances.json")
    meta = property(lambda self: self.root / "session.json")

    def read(self, path: Path) -> Any:
        return json.loads(path.read_text())

    def write(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(path)

    def keyframes(self) -> list[dict[str, Any]]:
        return self.read(self.keyframes_json)["keyframes"]

    def scene(self) -> str:
        return self.read(self.meta)["scene"] if self.meta.exists() else "home"
