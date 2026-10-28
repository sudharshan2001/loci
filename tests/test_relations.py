import numpy as np

from loci.stages.relations import is_inside, is_on, location_text, side_of, walk_metrics

UP = np.array([0.0, -1.0, 0.0])  # OpenCV world: y points down, so up is -y


def _obj(oid, label, lo, hi, area="office"):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return {
        "object_id": oid,
        "label": label,
        "box_min": lo.tolist(),
        "box_max": hi.tolist(),
        "centroid": ((lo + hi) / 2).tolist(),
        "area": area,
    }


DESK = _obj("desk_1", "desk", [-0.6, 0.0, 1.0], [0.6, 0.75, 1.6])  # top surface at y = 0.0 (height 0)


def test_on_desk() -> None:
    mug = _obj("mug_1", "mug", [0.0, -0.1, 1.2], [0.08, 0.0, 1.28])  # bottom at desk top
    assert is_on(mug, DESK, UP)


def test_floating_and_beside_not_on() -> None:
    lamp = _obj("lamp_1", "lamp", [0.0, -1.0, 1.2], [0.1, -0.6, 1.3])  # 0.6 m above
    beside = _obj("bag_1", "bag", [1.0, -0.1, 1.2], [1.2, 0.0, 1.4])  # outside footprint
    assert not is_on(lamp, DESK, UP) and not is_on(beside, DESK, UP)


def test_inside() -> None:
    drawer = _obj("drawer_1", "drawer", [0, 0, 0], [0.5, 0.2, 0.5])
    keys = _obj("keys_1", "keys", [0.2, 0.05, 0.2], [0.25, 0.08, 0.25])
    assert is_inside(keys, drawer)


def test_side_of_from_camera() -> None:
    cam = np.eye(4)  # at origin looking +z, x to the right
    assert side_of(np.array([-0.5, 0, 2]), np.array([0.5, 0, 2]), cam) == "left_of"
    assert side_of(np.array([0.5, 0, 2]), np.array([-0.5, 0, 2]), cam) == "right_of"


def test_location_text_prefers_on() -> None:
    mug = _obj("mug_1", "mug", [0, 0, 0], [0.1, 0.1, 0.1])
    rels = [
        {"subject": "mug_1", "relation": "near", "object": "desk_1", "dist_m": 0.3},
        {"subject": "mug_1", "relation": "on", "object": "desk_1", "dist_m": 0.3},
    ]
    assert location_text(mug, rels, {"desk_1": DESK}) == "on the desk, in the office"


def test_walk_metrics_side_and_distance() -> None:
    # walk straight along +z (forward), 1 m per frame; with up = -y, +x is to the right
    poses = np.stack([np.eye(4) for _ in range(11)])
    poses[:, 2, 3] = np.arange(11)
    shop_r = _obj("sign_1", "shop sign", [3, -3, 4.9], [4, -2, 5.1])
    shop_l = _obj("sign_2", "shop sign", [-4, -3, 7.9], [-3, -2, 8.1])
    walk_metrics([shop_r, shop_l], poses, [float(i) for i in range(11)], UP)
    assert shop_r["side"] == "right" and shop_r["distance_along_walk_m"] == 5.0 and shop_r["passed_at"] == 5.0
    assert shop_l["side"] == "left" and shop_l["distance_along_walk_m"] == 8.0


def test_far_object_is_placed_from_its_sighting() -> None:
    # walk 10 m along +z; a sign seen at frame 3 whose depth put it 80 m away (bad depth)
    poses = np.stack([np.eye(4) for _ in range(11)])
    poses[:, 2, 3] = np.arange(11)
    sign = _obj("sign_1", "shop sign", [4, -3, 79], [5, -2, 81])
    sign["best_obs"] = "obs_3"
    obs = {"obs_3": {"keyframe": 3, "centroid": [4.5, -2.5, 80.0]}}
    walk_metrics([sign], poses, [float(i) for i in range(11)], UP, obs, {k: k for k in range(11)}, max_depth=15)
    assert sign["side"] == "right" and sign["distance_along_walk_m"] <= 10 and sign["distance_along_walk_m"] >= 3
