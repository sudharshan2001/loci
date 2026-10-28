import numpy as np

from loci.register import apply, fit_yaw_similarity, register


def _objs(points: dict[str, list[list[float]]]) -> list[dict]:
    return [{"label": lab, "centroid": p} for lab, ps in points.items() for p in ps]


def _transform(objs: list[dict], yaw: float, t: np.ndarray, scale: float = 1.0) -> list[dict]:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return [{**o, "centroid": (scale * R @ np.asarray(o["centroid"]) + t).tolist()} for o in objs]


def test_fit_yaw_similarity_recovers_transform() -> None:
    rng = np.random.default_rng(0)
    P = rng.uniform(-2, 2, (6, 3))
    th, t = 0.7, np.array([1.0, 0.2, -3.0])
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    s, R2, t2 = fit_yaw_similarity(P, P @ R.T * 1.1 + t)
    assert abs(s - 1.1) < 1e-6 and np.allclose(R2, R, atol=1e-6) and np.allclose(t2, t, atol=1e-6)


def test_register_with_moved_and_extra_objects() -> None:
    room = {
        "desk": [[2.0, 0.6, 1.0]],
        "gas stove": [[-1.0, 0.5, 3.0]],
        "door": [[0.0, 0.0, -1.0], [3.5, 0.0, 2.5]],
        "bed": [[4.0, 0.7, -0.5]],
    }
    morning = _objs(room)
    evening = _transform(morning, yaw=1.2, t=np.array([0.5, 0.0, 2.0]))
    evening[0]["centroid"][0] += 0.05  # small measurement noise
    evening.append({"label": "keys", "centroid": [9.0, 0.0, 9.0]})  # movable things are ignored
    res = register(morning, evening)
    assert res["ok"] and res["n_inliers"] >= 4 and res["rms_m"] < 0.1
    # the morning stove lands on the evening stove
    assert np.linalg.norm(apply(res["transform"], morning[1]["centroid"]) - evening[1]["centroid"]) < 0.1


def test_register_refuses_without_enough_anchors() -> None:
    res = register(_objs({"desk": [[0, 0, 0]], "keys": [[1, 0, 0]]}), _objs({"desk": [[0, 0, 0]]}))
    assert not res["ok"]
