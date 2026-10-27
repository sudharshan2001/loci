import numpy as np

from loci.masks import decode, encode, iou, suppress_across_labels


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((50, 60), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_rle_roundtrip() -> None:
    m = _box(5, 20, 10, 40)
    assert np.array_equal(decode(encode(m)), m)


def test_iou() -> None:
    assert iou(_box(0, 10, 0, 10), _box(0, 10, 0, 10)) == 1.0
    assert iou(_box(0, 10, 0, 10), _box(20, 30, 20, 30)) == 0.0


def test_suppress_keeps_best_and_records_alias() -> None:
    dets = [
        {"label": "computer monitor", "score": 0.9, "mask": _box(0, 20, 0, 20)},
        {"label": "monitor", "score": 0.7, "mask": _box(0, 20, 0, 21)},
        {"label": "keyboard", "score": 0.8, "mask": _box(30, 40, 0, 20)},
    ]
    kept = suppress_across_labels(dets)
    assert [k["label"] for k in kept] == ["computer monitor", "keyboard"]
    assert kept[0]["aliases"] == ["monitor"]


def test_keys_on_keyboard_dropped_but_keys_on_desk_kept() -> None:
    from loci.masks import drop_confused_parts

    dets = [
        {"label": "silver laptop", "score": 0.9, "mask": _box(0, 30, 0, 40)},
        {"label": "keys", "score": 0.6, "mask": _box(10, 15, 10, 15)},  # keyboard keys on the laptop
        {"label": "keys", "score": 0.7, "mask": _box(40, 45, 45, 50)},  # real keys elsewhere
    ]
    kept = drop_confused_parts(dets)
    assert [d["label"] for d in kept] == ["silver laptop", "keys"]
    assert kept[1]["mask"][42, 47]
