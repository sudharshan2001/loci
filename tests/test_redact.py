import cv2
import numpy as np

from loci.stages.redact import blur_regions


def _text_image() -> np.ndarray:
    img = np.full((240, 320, 3), 255, np.uint8)
    cv2.putText(img, "SECRET 1234", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


def test_empty_mask_is_identity() -> None:
    img = _text_image()
    assert np.array_equal(blur_regions(img, np.zeros(img.shape[:2], bool)), img)


def test_masked_text_loses_detail() -> None:
    img = _text_image()
    mask = np.zeros(img.shape[:2], bool)
    mask[90:150, 30:300] = True
    out = blur_regions(img, mask)
    lap = lambda a: cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)[90:150, 30:300], cv2.CV_64F).var()  # noqa: E731
    assert lap(out) < 0.02 * lap(img)


def test_far_pixels_untouched() -> None:
    img = _text_image()
    img[200:, :] = 50
    mask = np.zeros(img.shape[:2], bool)
    mask[90:150, 30:300] = True
    out = blur_regions(img, mask)
    assert np.array_equal(out[220:, :], img[220:, :])
