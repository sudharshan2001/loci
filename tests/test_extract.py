import cv2
import numpy as np

from loci.stages.extract import select_keyframes, sharpness


def test_sharpness_prefers_edges() -> None:
    flat = np.full((240, 320), 128, np.uint8)
    checker = (np.indices((240, 320)).sum(0) // 8 % 2 * 255).astype(np.uint8)
    assert sharpness(checker) > sharpness(flat)


def test_select_caps_count_and_spreads() -> None:
    times = np.arange(100) / 4.0
    scores = np.ones(100)
    picked = select_keyframes(times, scores, max_keyframes=10, min_quantile=0.3)
    assert len(picked) == 10
    gaps = np.diff(times[picked])
    assert gaps.max() < 2 * gaps.min() + 1e-6


def test_select_picks_sharpest_in_bin() -> None:
    times = np.array([0.0, 0.1, 0.2, 1.0, 1.1, 1.2])
    scores = np.array([1.0, 9.0, 2.0, 3.0, 1.0, 8.0])
    picked = select_keyframes(times, scores, max_keyframes=2, min_quantile=0.0)
    assert picked == [1, 5]


def test_select_empty() -> None:
    assert select_keyframes(np.array([]), np.array([]), 10, 0.3) == []


def test_progress_puts_more_keyframes_in_fast_segments() -> None:
    from loci.stages.extract import progress

    times = np.arange(100) / 4.0  # 25 s
    shifts = np.full(100, 0.01)
    shifts[50:60] = 0.3  # a fast pan in the middle
    picked = select_keyframes(times, np.ones(100), 20, 0.0, progress(times, shifts))
    in_pan = sum(50 <= i < 60 for i in picked)
    assert in_pan >= 4  # time-only selection would give about 2


def test_image_shift_detects_translation() -> None:
    from loci.stages.extract import image_shift

    rng = np.random.default_rng(0)
    img = (rng.random((120, 160)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    moved = np.roll(img, 16, axis=1)
    assert image_shift(img, img) < 0.02
    assert 0.07 < image_shift(img, moved) < 0.15
