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


def test_shop_open_only_on_the_street() -> None:
    assert "shop_open" in question_for("fruit stand", "street")[1]
    assert "shop_open" in question_for("coconut cart", "street")[1]
    assert question_for("monitor stand", "home") is None or "shop_open" not in question_for("monitor stand", "home")[1]


def test_quick_pass_then_budgeted_second_looks(tmp_path, monkeypatch) -> None:
    import json

    import cv2

    from loci.providers.base import VisionAnswer
    from loci.session import Session
    from loci.stages import attributes as A

    s = Session(tmp_path / "walk")
    (s.root / "keyframes").mkdir(parents=True)
    cv2.imwrite(str(s.keyframes_dir / "0000.jpg"), np.zeros((100, 100, 3), np.uint8))
    s.meta.write_text(json.dumps({"scene": "street"}))
    obs = [
        {"obs_id": f"obs_{i}", "keyframe": 0, "t": float(i), "bbox": [0, 0, 10 * (i + 1), 10 * (i + 1)], "quality": 1}
        for i in range(3)
    ]
    s.observations.write_text(json.dumps(obs))
    objs = [{"object_id": f"sign_{i}", "label": "signboard", "obs_ids": [f"obs_{i}"]} for i in range(3)]
    objs.append({"object_id": "lamp_1", "label": "street lamp", "obs_ids": ["obs_0"]})
    s.objects.write_text(json.dumps(objs))
    monkeypatch.setattr(
        A,
        "settings",
        lambda scene: {"per_object": 1, "skip": {"street lamp"}, "second_budget_s": 60.0, "min_confidence": 0.7},
    )

    calls = []

    class Fake:
        def check(self, image, question, tight=None, wide=None, mode="auto"):
            calls.append((mode, image.name.split("_ctx")[0]))
            if mode == "quick" or image.name.startswith("obs_0"):
                return VisionAnswer({"text": "?", "readable": True}, 0.4, False, ["glimmer:low"])
            return VisionAnswer({"text": "BHIMA", "readable": True}, 0.9, False, ["glimmer:high"])

    assert A.pending(s) == 3  # the lamp is skipped
    res = A.run(s, Fake())
    # every sign gets a quick look, then the unsure ones a second look, largest first
    assert calls == [("quick", "obs_0"), ("quick", "obs_1"), ("quick", "obs_2")] + [
        ("second", "obs_2"),
        ("second", "obs_1"),
        ("second", "obs_0"),
    ]
    best = {r["object_id"]: r for r in res}
    assert best["sign_2"]["answer"]["text"] == "BHIMA"
    assert best["sign_2"]["attempts"] == ["glimmer:low", "glimmer:high"]
    assert best["sign_0"]["confidence"] == 0.4 and "lamp_1" not in best
    assert best["sign_0"]["attempts"] == ["glimmer:low", "glimmer:low"]
    assert A.pending(s) == 0
