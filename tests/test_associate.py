from loci.stages.associate import associate, box_iou


def _ob(obs_id: str, label: str, c, half=0.1, t=0.0, score=0.9, aliases=None) -> dict:
    return {
        "obs_id": obs_id,
        "label": label,
        "aliases": aliases or [],
        "centroid": list(c),
        "box_min": [x - half for x in c],
        "box_max": [x + half for x in c],
        "t": t,
        "score": score,
        "quality": score,
        "area_label": "office",
    }


def test_box_iou() -> None:
    assert box_iou([0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]) == 1.0
    assert box_iou([0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]) == 0.0


def test_same_label_close_merges_far_splits() -> None:
    obs = [
        _ob("a", "potted plant", [0, 0, 0], t=1),
        _ob("b", "potted plant", [0.05, 0, 0.02], t=2),
        _ob("c", "potted plant", [2.0, 0, 0], t=3),
    ]
    objs = associate(obs)
    assert len(objs) == 2
    first = next(o for o in objs if "a" in o["obs_ids"])
    assert first["obs_ids"] == ["a", "b"] and first["first_seen"] == 1 and first["last_seen"] == 2


def test_different_labels_do_not_merge_unless_alias() -> None:
    obs = [_ob("a", "keyboard", [0, 0, 0]), _ob("b", "laptop", [0, 0, 0])]
    assert len(associate(obs)) == 2
    obs = [_ob("a", "computer monitor", [0, 0, 0], aliases=["monitor"]), _ob("b", "monitor", [0.02, 0, 0])]
    objs = associate(obs)
    assert len(objs) == 1 and objs[0]["label"] == "computer monitor"


def test_ids_are_stable_and_ordered_by_first_seen() -> None:
    obs = [_ob("x", "keys", [5, 0, 0], t=9), _ob("y", "keys", [0, 0, 0], t=1)]
    objs = associate(obs)
    assert [o["object_id"] for o in objs] == ["keys_1", "keys_2"]
    assert objs[0]["obs_ids"] == ["y"]
