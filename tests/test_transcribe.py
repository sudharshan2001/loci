from loci.stages.transcribe import link


def test_link_window_and_order() -> None:
    utts = [{"start_s": 10.0, "end_s": 12.0, "text": "water this one tomorrow"}]
    obs = [
        {"t": 11.0, "object_id": "potted_plant_1"},
        {"t": 13.5, "object_id": "desk_1"},
        {"t": 20.0, "object_id": "keys_1"},  # outside +/- 2 s
        {"t": 11.2, "object_id": "potted_plant_1"},
    ]
    links = link(utts, obs)
    assert [x["object_id"] for x in links] == ["potted_plant_1", "desk_1"]
