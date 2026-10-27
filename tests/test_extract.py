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
