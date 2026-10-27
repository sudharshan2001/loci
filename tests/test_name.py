from loci.stages.name import fill_nearest, normalise, spread


def test_normalise_cleans_and_dedupes() -> None:
    out = normalise(
        [
            "The Potted Plant",
            "potted plant",
            "a laptop!",
            "Person",
            "white wall",
            "very long phrase with many words",
        ]
    )
    assert out == ["potted plant", "laptop"]


def test_fill_nearest() -> None:
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert fill_nearest(times, {0: "kitchen", 5: "bedroom"}) == ["kitchen"] * 3 + ["bedroom"] * 3


def test_spread_covers_ends() -> None:
    assert spread(80, 8)[0] == 0 and spread(80, 8)[-1] == 79
    assert spread(3, 8) == [0, 1, 2]
