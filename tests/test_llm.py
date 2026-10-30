import pytest

from loci.llm import extract_json


def test_plain_array() -> None:
    assert extract_json('["desk", "plant"]') == ["desk", "plant"]


def test_fenced_object_with_prose() -> None:
    text = 'Here you go:\n```json\n{"burner_lit": true, "confidence": 0.9}\n```\nDone.'
    assert extract_json(text) == {"burner_lit": True, "confidence": 0.9}


def test_no_json_raises() -> None:
    with pytest.raises(ValueError):
        extract_json("I could not tell.")


def test_image_part_downscales_large_frames(tmp_path) -> None:
    import base64

    import cv2
    import numpy as np

    from loci.llm import image_part

    big = tmp_path / "big.jpg"
    cv2.imwrite(str(big), np.zeros((1080, 1920, 3), np.uint8))
    url = image_part(big)["image_url"]["url"]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 1024
