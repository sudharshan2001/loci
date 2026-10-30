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


def test_overlapping_partial_views_merge_and_weak_singletons_drop() -> None:
    # two partial views of one bed, far enough apart to start separate objects, but overlapping boxes
    obs = [
        _ob("a", "bed", [0, 0, 0], half=0.8, t=1),
        _ob("b", "bed", [1.2, 0, 0], half=0.8, t=2),
        _ob("c", "vase", [5, 0, 0], score=0.4, t=3),  # single weak sighting
    ]
    objs = associate(obs)
    assert [o["label"] for o in objs] == ["bed"] and objs[0]["n_obs"] == 2


def test_two_boards_in_one_frame_stay_separate() -> None:
    a = {**_ob("o1", "signboard", [0, 0, 0], half=1.0), "keyframe": 5}
    b = {**_ob("o2", "signboard", [0.5, 0, 0], half=1.0), "keyframe": 5}
    c = {**_ob("o3", "signboard", [0.1, 0, 0], half=1.0), "keyframe": 6}
    objs = associate([a, b, c])
    assert sorted(len(o["obs_ids"]) for o in objs) == [1, 2]
    # other things seen twice in a frame (partial views) still merge
    d = {**_ob("o4", "desk", [0, 0, 0], half=1.0), "keyframe": 5}
    e = {**_ob("o5", "desk", [0.5, 0, 0], half=1.0), "keyframe": 5}
    assert len(associate([d, e])) == 1
