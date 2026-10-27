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
