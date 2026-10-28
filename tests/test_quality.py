import numpy as np

from loci.quality import coach, reconstruction, spans


def test_smooth_walk_is_good_and_jumps_are_broken() -> None:
    times = np.arange(40) * 0.25
    walk = np.stack([np.zeros(40), np.zeros(40), times * 1.0], 1)  # 1 m/s
    assert reconstruction(times, walk, "home")["verdict"] == "good"
    jumpy = walk.copy()
    jumpy[10:20:2, 0] += 1.5  # 6 m/s back and forth
    r = reconstruction(times, jumpy, "home")
    assert r["verdict"] == "broken" and r["bad_spans"][0][0] >= 2.0


def test_spans_merge_close_steps() -> None:
    flags = np.array([False, True, True, False, False, False, False, True])
    times = np.arange(9, dtype=float)
    assert spans(flags, times, min_gap_s=1.0) == [(1.0, 3.0), (7.0, 8.0)]


def test_coach_advice() -> None:
    kf = {"duration_s": 18.0, "view_motion": 19.0, "keyframes": [{"sharpness": 300.0}] * 60}
    recon = {"verdict": "broken", "bad_spans": [(5.0, 8.0)]}
    attrs = [{"kind": "sign", "answer": {"readable": False}}] * 3 + [{"kind": "sign", "answer": {"readable": True}}]
    tips = coach({"scene": "home"}, kf, recon, attrs, {"utterances": []}, 848, 478)
    text = " ".join(tips)
    assert "0:05" in text and "1080p" in text and "3 of 4" in text and "spoken" in text
    assert "screen-widths" in text
