"""Capture quality: is the 3D trustworthy, and what should be re-walked? (the capture coach)

Pure functions over data the pipeline already stores; no models.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# fastest believable camera speed between keyframes (m/s): walking pace plus margin
MAX_SPEED = {"home": 2.0, "street": 4.0}
# keyframes should overlap: the view should move at most this many image widths between them
MAX_SHIFT_PER_KEYFRAME = 0.2


def fmt(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def spans(flags: np.ndarray, times: np.ndarray, min_gap_s: float = 1.0) -> list[tuple[float, float]]:
    """Merge flagged keyframe steps into time spans [(start, end)]."""
    out: list[list[float]] = []
    for i in np.flatnonzero(flags):
        t0, t1 = float(times[i]), float(times[i + 1])
        if out and t0 - out[-1][1] <= min_gap_s:
            out[-1][1] = t1
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def reconstruction(times: np.ndarray, cams: np.ndarray, scene: str) -> dict[str, Any]:
    """Judge camera motion: jumps faster than a person can move mean the 3D broke there."""
    dt = np.maximum(np.diff(times), 1e-3)
    speed = np.linalg.norm(np.diff(cams, axis=0), axis=1) / dt
    bad = speed > MAX_SPEED.get(scene, 2.0)
    duration = float(times[-1] - times[0]) or 1.0
    path = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum())
    frac = float(bad.mean()) if len(bad) else 0.0
    # home capture with 9% impossible jumps already showed ghosted walls and duplicated objects
    verdict = "good" if frac < 0.03 else "unreliable" if frac < 0.08 else "broken"
    return {
        "verdict": verdict,
        "bad_step_fraction": round(frac, 3),
        "max_speed_mps": round(float(speed.max()) if len(speed) else 0.0, 2),
        "mean_speed_mps": round(path / duration, 2),
        "bad_spans": [(round(a, 1), round(b, 1)) for a, b in spans(bad, times)],
    }


def coach(
    meta: dict,
    keyframes: dict,
    recon: dict | None,
    attributes: list[dict],
    utterances: dict | None,
    width: int,
    height: int,
) -> list[str]:
    """Plain advice for the next capture, most important first."""
    tips: list[str] = []
    scene = meta.get("scene", "home")
    if recon and recon["verdict"] != "good":
        where = ", ".join(f"{fmt(a)}–{fmt(b)}" for a, b in recon["bad_spans"][:3])
        tips.append(
            f"The 3D map is {recon['verdict']}: the camera jumped faster than you can walk around {where}. "
            "Re-walk those parts slowly; turn over 2–3 seconds instead of whipping round."
        )
    n = len(keyframes["keyframes"])
    motion = keyframes.get("view_motion")
    if motion and motion / max(n, 1) > MAX_SHIFT_PER_KEYFRAME:
        need_s = keyframes["duration_s"] * motion / (n * MAX_SHIFT_PER_KEYFRAME)
        tips.append(
            f"The view moved {motion:.0f} screen-widths in {keyframes['duration_s']:.0f} s, too fast for "
            f"neighbouring frames to overlap. Take about {need_s:.0f} s for the same walk."
        )
    if max(width, height) < 1600:
        tips.append(f"Video is {width}x{height}; record in 1080p so signs and small items are readable.")
    if height > width:
        tips.append("Video is portrait; landscape sees more of the room or street at once.")
    signs = [a for a in attributes if a.get("kind") == "sign"]
    unread = [a for a in signs if not a["answer"].get("readable", False)]
    if signs and len(unread) / len(signs) > 0.3:
        tips.append(
            f"{len(unread)} of {len(signs)} shop signs could not be read: slow down and face each sign for 1–2 seconds."
        )
    if scene == "home" and utterances is not None and not utterances.get("utterances"):
        tips.append("No spoken notes were heard. Say a note out loud near an object, e.g. 'water this plant tomorrow'.")
    blurry = sum(k["sharpness"] < 50 for k in keyframes["keyframes"])
    if blurry > 0.2 * n:
        tips.append(f"{blurry} of {n} keyframes are blurry: hold the phone steadier or add light.")
    return tips
