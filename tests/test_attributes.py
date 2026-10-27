import numpy as np

from loci.stages.attributes import crop, question_for


def test_question_routing() -> None:
    assert question_for("shop sign")[0] == "sign"
    assert question_for("signboard")[0] == "sign"
    assert question_for("gas stove")[0] == "state"
    assert "burner_lit" in question_for("burner flame")[1]
    assert "screen_on" in question_for("computer monitor")[1]
    assert question_for("potted plant") is None


def test_crop_clamps_to_image() -> None:
    img = np.zeros((100, 200, 3), np.uint8)
    assert crop(img, [0, 0, 50, 50], 1.0).shape == (100, 100, 3)
    assert crop(img, [150, 50, 200, 100], 0.0).shape == (50, 50, 3)
