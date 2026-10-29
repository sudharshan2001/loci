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


def image_shift(a: np.ndarray, b: np.ndarray, width: int = 160) -> float:
    """How far the view moved between two frames, as a fraction of the image width (phase correlation)."""

    def prep(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, (width, max(1, round(g.shape[0] * width / g.shape[1]))))
        return g.astype(np.float32)

    ga, gb = prep(a), prep(b)
    (dx, dy), response = cv2.phaseCorrelate(ga, gb)
    # a weak correlation peak means the frames barely overlap: count it as a large move
    shift = np.hypot(dx, dy) / width
    return float(max(shift, 0.5 * (1 - min(1.0, response * 4))))


def progress(times: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Cumulative 'how much changed': half elapsed time, half view motion, each normalised to 1."""
    t = (times - times[0]) / max(1e-6, times[-1] - times[0])
    m = np.r_[0, np.cumsum(shifts[1:])]
    m = m / m[-1] if m[-1] > 0 else t
    return 0.5 * t + 0.5 * m


def select_keyframes(
    times: np.ndarray,
    scores: np.ndarray,
    max_keyframes: int,
    min_quantile: float,
    prog: np.ndarray | None = None,
) -> list[int]:
    """Pick at most one candidate per equal bin of progress (default: time): the sharpest above the blur floor.

    With `prog` from `progress`, fast pans get more keyframes so neighbours still overlap for 3D.
    Returns indices into `times`, in time order. A bin whose best candidate is blurry stays empty unless
    that would leave a gap; then the sharpest one is kept anyway.
    """
    if len(times) == 0:
        return []
    axis = times if prog is None else prog
    floor = float(np.quantile(scores, min_quantile)) if len(scores) > 1 else -np.inf
    n_bins = min(max_keyframes, len(times))
    edges = np.linspace(axis[0], axis[-1] + 1e-6, n_bins + 1)
    picked: list[int] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = np.flatnonzero((axis >= lo) & (axis < hi))
        if len(idx) == 0:
            continue
        best = int(idx[np.argmax(scores[idx])])
        if scores[best] >= floor or not picked or axis[best] - axis[picked[-1]] > 2 * (hi - lo):
            picked.append(best)
    return picked


def keyframe_budget(duration_s: float, per_second: float = 1.0, lo: int = 60, hi: int = 360) -> int:
    """About one keyframe per second of walking, at least 60 (short clips) and at most 360 (6-minute walks)."""
    return int(np.clip(round(duration_s * per_second), lo, hi))


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
    if max_keyframes <= 0:  # budget follows the length of the walk
        max_keyframes = keyframe_budget(n_frames / fps)

    # pass 1 (sequential read; seeking every candidate is slow on long 1080p videos): score candidates
    # (sharpness) and measure how far the view moved between them
    cand_frames, cand_scores, shifts = [], [], []
    prev = None
    for fi in range(n_frames):
        if not cap.grab():
            break
        if fi % step:
            continue
        ok, img = cap.retrieve()
        if not ok:
            break
        cand_frames.append(fi)
        cand_scores.append(sharpness(img))
        shifts.append(0.0 if prev is None else image_shift(prev, img))
        prev = img
    frames = np.array(cand_frames)
    scores = np.array(cand_scores)
    times = frames / fps
    picked = select_keyframes(times, scores, max_keyframes, min_quantile, progress(times, np.array(shifts)))

    # pass 2 (sequential again): write the picked frames at full resolution
    wanted = {int(frames[ci]): ci for ci in picked}
    keyframes: list[Keyframe] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last = max(wanted) if wanted else -1
    for fi in range(last + 1):
        if not cap.grab():
            break
        if fi not in wanted:
            continue
        ok, img = cap.retrieve()
        if not ok:
            continue
        ci = wanted[fi]
        k = len(keyframes)
        name = f"{k:04d}.jpg"
        cv2.imwrite(str(raw_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        keyframes.append(Keyframe(k, fi, round(float(times[ci]), 3), round(float(scores[ci]), 1), name))
    cap.release()

    has_audio = extract_audio(video, session_dir / "audio.wav")
    meta = {
        "video": str(video),
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 2),
        "n_candidates": len(frames),
        "view_motion": round(float(np.sum(shifts)), 2),  # total image widths the view moved
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
