import numpy as np

from sathealth.decision.events import build_events, outside_range, severity


def test_severity_levels() -> None:
    assert [severity(v, 4.5, 6.0) for v in (3.0, 5.0, 7.0)] == ["watch", "warning", "critical"]


def test_build_events() -> None:
    surprise = np.array([0, 3.5, 7.0, 3.0, 0, 0, 4.0, 4.6, 0])
    mask = surprise >= 3
    values = np.array([[0.5]] * 9)
    values[2, 0] = 9.0  # out of range inside the first event
    oor = outside_range(values, np.array([0.0]), np.array([1.0]), 0.05)
    events = build_events(
        mask,
        surprise,
        {"lstm": surprise, "transformer": surprise / 2},
        oor,
        warning=4.5,
        critical=6,
    )
    assert [(e.start, e.end, e.peak_step) for e in events] == [(1, 3, 2), (6, 7, 7)]
    assert [e.severity for e in events] == ["critical", "warning"]
    assert [e.kind for e in events] == ["point", "contextual"]
    assert events[0].contributions == {"lstm": 7.0, "transformer": 3.5}
    assert events[1].to_dict()["peak_surprise"] == 4.6
