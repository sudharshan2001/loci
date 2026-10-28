"""Stage 1: decode video, pick sharp, evenly spread keyframes, extract audio.

Raw (unredacted) keyframes go to <session>/raw_tmp/ only; the redact stage deletes them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Keyframe:
    index: int  # keyframe index within the session
    frame: int  # source video frame number
    t: float  # timestamp in seconds
    sharpness: float  # variance of Laplacian on a 640px-wide grey copy
    file: str  # file name inside keyframes/ (and raw_tmp/)


def sharpness(img: np.ndarray, width: int = 640) -> float:
    """Variance of Laplacian, measured at a fixed width so it is comparable across videos."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if grey.shape[1] != width:
        grey = cv2.resize(grey, (width, round(grey.shape[0] * width / grey.shape[1])))
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def select_keyframes(times: np.ndarray, scores: np.ndarray, max_keyframes: int, min_quantile: float) -> list[int]:
    """Pick at most one candidate per equal time bin: the sharpest, if above the blur floor.

    Returns indices into `times`, in time order. Bins whose best candidate is blurry stay empty
    unless the bin would otherwise leave a gap; then the sharpest one is kept anyway.
    """
    if len(times) == 0:
        return []
    floor = float(np.quantile(scores, min_quantile)) if len(scores) > 1 else -np.inf
    n_bins = min(max_keyframes, len(times))
    edges = np.linspace(times[0], times[-1] + 1e-6, n_bins + 1)
    picked: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = np.flatnonzero((times >= lo) & (times < hi))
        if len(idx) == 0:
            continue
        best = int(idx[np.argmax(scores[idx])])
        if scores[best] >= floor or not picked or times[best] - times[picked[-1]] > 2 * (hi - lo):
            picked.append(best)
    return picked


def session_id_for(video: Path) -> str:
    """Stable id: file stem plus a short hash of size and first MiB."""
    h = hashlib.sha1(str(video.stat().st_size).encode())
    with open(video, "rb") as f:
        h.update(f.read(1 << 20))
    return f"{video.stem}_{h.hexdigest()[:8]}"


def captured_at(video: Path) -> str:
    """When the video was recorded: its creation_time tag, else the file's modification time (ISO 8601)."""
    from datetime import datetime

    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags=creation_time",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    tag = res.stdout.strip()
    if tag:
        return tag.replace("Z", "+00:00")
    return datetime.fromtimestamp(video.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def extract_audio(video: Path, out: Path) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.warning("no audio extracted: %s", res.stderr.strip()[:200])
        return False
    return True


def run(
    video: Path, session_dir: Path, max_keyframes: int, candidate_fps: float, min_quantile: float
) -> list[Keyframe]:
    session_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = session_dir / "raw_tmp"
    raw_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / candidate_fps))

    # pass 1: score candidates
    cand_frames, cand_scores = [], []
    for fi in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            break
        cand_frames.append(fi)
        cand_scores.append(sharpness(img))
    frames = np.array(cand_frames)
    scores = np.array(cand_scores)
    times = frames / fps
    picked = select_keyframes(times, scores, max_keyframes, min_quantile)

    # pass 2: write the picked frames at full resolution
    keyframes: list[Keyframe] = []
    for k, ci in enumerate(picked):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[ci]))
        ok, img = cap.read()
        if not ok:
            continue
        name = f"{k:04d}.jpg"
        cv2.imwrite(str(raw_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        keyframes.append(Keyframe(k, int(frames[ci]), round(float(times[ci]), 3), round(float(scores[ci]), 1), name))
    cap.release()

    has_audio = extract_audio(video, session_dir / "audio.wav")
    meta = {
        "video": str(video),
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "n_candidates": len(frames),
        "has_audio": has_audio,
        "keyframes": [asdict(k) for k in keyframes],
    }
    (session_dir / "keyframes.json").write_text(json.dumps(meta, indent=2))
    gaps = np.diff([k.t for k in keyframes]) if len(keyframes) > 1 else np.array([0.0])
    log.info(
        "%s: %d/%d candidates kept, max gap %.2fs, median sharpness %.0f",
        video.name,
        len(keyframes),
        len(frames),
        gaps.max(),
        np.median([k.sharpness for k in keyframes]) if keyframes else 0,
    )
    return keyframes
