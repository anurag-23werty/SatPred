import sqlite3
from pathlib import Path

import pytest

from sathealth.decision.events import Event
from sathealth.service.store import Store


def _event(start: int = 10, severity: str = "watch") -> Event:
    return Event(start, start + 5, start + 2, 3.5, severity, "contextual", {"lstm": 4.0})  # type: ignore[arg-type]


def test_event_lifecycle_and_filters(tmp_path: Path) -> None:
    store = Store(tmp_path / "s.db")
    replay = store.start_replay("fused", ["A-1"])
    event_id = store.open_event(replay, "A-1", "SMAP", _event())
    assert store.event(event_id)["status"] == "open"
    final = Event(10, 30, 20, 7.0, "critical", "point", {"lstm": 7.0})
    assert store.close_event(replay, "A-1", final) == event_id
    row = store.event(event_id)
    assert (row["status"], row["end"], row["severity"]) == ("closed", 30, "critical")
    assert row["contributions"] == {"lstm": 7.0}
    store.open_event(replay, "B-1", "MSL", _event(50))
    assert [e["chan_id"] for e in store.events(replay)] == ["A-1", "B-1"]
    assert [e["chan_id"] for e in store.events(replay, status="open")] == ["B-1"]
    assert store.events(replay, severity="critical", chan_id="A-1")[0]["id"] == event_id
    assert store.event(999) is None
    with pytest.raises(KeyError, match="no open event"):
        store.close_event(replay, "Z-9", _event())
    store.finish_replay(replay)
    store.close()


def test_verdicts_feedback_and_explanation_cache(tmp_path: Path) -> None:
    store = Store(tmp_path / "s.db")
    replay = store.start_replay("fused", ["A-1"])
    a = store.open_event(replay, "A-1", "SMAP", _event(10, "critical"))
    b = store.open_event(replay, "A-1", "SMAP", _event(40, "watch"))
    store.add_verdict(a, "false_alarm", "op1")
    store.add_verdict(a, "confirmed", "op2", "on second look")  # latest verdict counts
    store.add_verdict(b, "unsure", "op1")
    summary = store.feedback_summary()
    assert summary["critical"] == {
        "reviewed": 1,
        "confirmed": 1,
        "false_alarm": 0,
        "precision": 1.0,
    }
    assert summary["all"]["reviewed"] == 2
    assert summary["watch"]["precision"] is None, "unsure verdicts do not count"
    assert store.event(a)["verdict"] == "confirmed"
    assert [v["operator"] for v in store.verdicts(a)] == ["op1", "op2"]
    assert len(store.verdicts()) == 3
    with pytest.raises(ValueError, match="verdict"):
        store.add_verdict(a, "maybe", "op1")
    with pytest.raises(KeyError):
        store.add_verdict(999, "confirmed", "op1")
    assert store.explanation(a) is None
    store.save_explanation(a, {"sentences": []})
    assert store.explanation(a) == {"sentences": []}


def test_schema_version_is_checked(tmp_path: Path) -> None:
    path = tmp_path / "s.db"
    Store(path).close()
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        Store(path)
