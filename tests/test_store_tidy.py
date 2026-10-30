from loci.stages.store import tidy_objects


def _o(oid, label, text=None):
    return {"object_id": oid, "label": label, "aliases": ["x"], "text": text, "text_english": text}


def test_tidy_drops_carried_things_and_keeps_places() -> None:
    objs = [_o("a", "backpack"), _o("b", "shopping bag"), _o("c", "yellow green auto rickshaw"), _o("d", "bus stop")]
    kept = tidy_objects(objs, ["backpack", "bag", "auto rickshaw", "bus"])
    assert [o["object_id"] for o in kept] == ["d"]


def test_tidy_brand_labels_and_fragments() -> None:
    kept = tidy_objects([_o("a", "levi s outlet sign", "megamart"), _o("b", "shop sign", "S")], [])
    assert kept[0]["label"] == "signboard" and kept[0]["text"] == "megamart" and kept[0]["aliases"] == []
    assert kept[1]["label"] == "shop sign" and kept[1]["text"] is None
    # vowel signs count as letters: a three-letter Kannada name is a real reading
    assert tidy_objects([_o("c", "signboard", "ಭೀಮ")], [])[0]["text"] == "ಭೀಮ"
