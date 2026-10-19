"""SQLite persistence for replays, events, explanations and operator verdicts.

One connection in WAL mode, guarded by a lock: the API serves requests from a thread pool
while the replay writes events, and SQLite allows one writer at a time anyway. All SQL lives
here; the rest of the service works with plain dictionaries.
"""

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sathealth.decision.events import Event

SCHEMA_VERSION = 2
VERDICTS = ("confirmed", "false_alarm", "unsure")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS replays (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    decision TEXT NOT NULL,
    channels TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    replay_id INTEGER NOT NULL REFERENCES replays(id),
    chan_id TEXT NOT NULL,
    spacecraft TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
    start INTEGER NOT NULL,
    end_step INTEGER NOT NULL,
    peak_step INTEGER NOT NULL,
    peak_surprise REAL NOT NULL,
    severity TEXT NOT NULL,
    kind TEXT NOT NULL,
    contributions TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    UNIQUE (replay_id, chan_id, start)
);
CREATE INDEX IF NOT EXISTS events_by_replay ON events (replay_id, status);
CREATE TABLE IF NOT EXISTS explanations (
    event_id INTEGER PRIMARY KEY REFERENCES events(id),
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL REFERENCES events(id),
    verdict TEXT NOT NULL CHECK (verdict IN ('confirmed', 'false_alarm', 'unsure')),
    operator TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS verdicts_by_event ON verdicts (event_id);
"""

# Forward migrations: MIGRATIONS[v] upgrades a version-v database to v + 1.
MIGRATIONS = {
    1: """
CREATE TABLE IF NOT EXISTS rebaselines (
    id INTEGER PRIMARY KEY,
    replay_id INTEGER NOT NULL REFERENCES replays(id),
    chan_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    steps INTEGER NOT NULL,
    operator TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    closed_event_id INTEGER REFERENCES events(id),
    created_at TEXT NOT NULL
);
""",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._db.executescript(SCHEMA)  # idempotent DDL; executescript commits itself
            row = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            version = SCHEMA_VERSION if row is None else int(row["value"])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{path}: schema version {version} is newer than this code "
                    f"({SCHEMA_VERSION}); upgrade sathealth"
                )
            if row is None:  # a new database gets every table at once
                for script in MIGRATIONS.values():
                    self._db.executescript(script)
            for v in range(version, SCHEMA_VERSION):
                self._db.executescript(MIGRATIONS[v])
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def _query(self, sql: str, args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- replays -------------------------------------------------------------------------

    def start_replay(self, decision: str, channels: list[str]) -> int:
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO replays (started_at, decision, channels) VALUES (?, ?, ?)",
                (_now(), decision, json.dumps(channels)),
            )
            return int(cur.lastrowid or 0)

    def finish_replay(self, replay_id: int) -> None:
        with self._tx() as db:
            db.execute("UPDATE replays SET finished_at=? WHERE id=?", (_now(), replay_id))

    # --- events --------------------------------------------------------------------------

    def open_event(self, replay_id: int, chan_id: str, spacecraft: str, event: Event) -> int:
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO events (replay_id, chan_id, spacecraft, status, start, end_step, "
                "peak_step, peak_surprise, severity, kind, contributions, opened_at) "
                "VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    replay_id,
                    chan_id,
                    spacecraft,
                    event.start,
                    event.end,
                    event.peak_step,
                    event.peak_surprise,
                    event.severity,
                    event.kind,
                    json.dumps(event.contributions),
                    _now(),
                ),
            )
            return int(cur.lastrowid or 0)

    def close_event(self, replay_id: int, chan_id: str, event: Event) -> int:
        """Finalise the open event with this start (every closed event was opened first)."""
        with self._tx() as db:
            row = db.execute(
                "SELECT id FROM events WHERE replay_id=? AND chan_id=? AND start=?",
                (replay_id, chan_id, event.start),
            ).fetchone()
            values = (
                event.end,
                event.peak_step,
                event.peak_surprise,
                event.severity,
                event.kind,
                json.dumps(event.contributions),
                _now(),
            )
            if row is None:
                raise KeyError(f"no open event {chan_id}@{event.start} in replay {replay_id}")
            db.execute(
                "UPDATE events SET status='closed', end_step=?, peak_step=?, peak_surprise=?, "
                "severity=?, kind=?, contributions=?, closed_at=? WHERE id=?",
                (*values, row["id"]),
            )
            return int(row["id"])

    def event(self, event_id: int) -> dict[str, Any] | None:
        rows = self._query(
            "SELECT e.*, (SELECT verdict FROM verdicts v WHERE v.event_id=e.id "
            "ORDER BY v.id DESC LIMIT 1) AS verdict FROM events e WHERE e.id=?",
            (event_id,),
        )
        return _decode(rows[0]) if rows else None

    def events(
        self,
        replay_id: int | None = None,
        *,
        status: str | None = None,
        severity: str | None = None,
        chan_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        where, args = [], []
        for column, value in (
            ("replay_id", replay_id),
            ("status", status),
            ("severity", severity),
            ("chan_id", chan_id),
        ):
            if value is not None:
                where.append(f"e.{column}=?")
                args.append(value)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        rows = self._query(
            "SELECT e.*, (SELECT verdict FROM verdicts v WHERE v.event_id=e.id "
            f"ORDER BY v.id DESC LIMIT 1) AS verdict FROM events e {clause} "
            "ORDER BY e.peak_surprise DESC, e.id LIMIT ?",
            (*args, limit),
        )
        return [_decode(r) for r in rows]

    # --- explanations --------------------------------------------------------------------

    def explanation(self, event_id: int) -> dict[str, Any] | None:
        rows = self._query("SELECT body FROM explanations WHERE event_id=?", (event_id,))
        return json.loads(rows[0]["body"]) if rows else None

    def save_explanation(self, event_id: int, body: dict[str, Any]) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO explanations VALUES (?, ?, ?)",
                (event_id, json.dumps(body), _now()),
            )

    # --- re-baselining -------------------------------------------------------------------

    def add_rebaseline(
        self,
        replay_id: int,
        chan_id: str,
        step: int,
        steps: int,
        operator: str,
        note: str = "",
        closed_event_id: int | None = None,
    ) -> int:
        with self._tx() as db:
            cur = db.execute(
                "INSERT INTO rebaselines (replay_id, chan_id, step, steps, operator, note, "
                "closed_event_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (replay_id, chan_id, step, steps, operator, note, closed_event_id, _now()),
            )
            return int(cur.lastrowid or 0)

    def rebaselines(self, chan_id: str | None = None) -> list[dict[str, Any]]:
        if chan_id is None:
            return self._query("SELECT * FROM rebaselines ORDER BY id")
        return self._query("SELECT * FROM rebaselines WHERE chan_id=? ORDER BY id", (chan_id,))

    # --- verdicts ------------------------------------------------------------------------

    def add_verdict(self, event_id: int, verdict: str, operator: str, note: str = "") -> int:
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        with self._tx() as db:
            if db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone() is None:
                raise KeyError(f"no event {event_id}")
            cur = db.execute(
                "INSERT INTO verdicts (event_id, verdict, operator, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (event_id, verdict, operator, note, _now()),
            )
            return int(cur.lastrowid or 0)

    def verdicts(self, event_id: int | None = None) -> list[dict[str, Any]]:
        if event_id is None:
            return self._query("SELECT * FROM verdicts ORDER BY id")
        return self._query("SELECT * FROM verdicts WHERE event_id=? ORDER BY id", (event_id,))

    def feedback_summary(self) -> dict[str, Any]:
        """Operator-confirmed precision by severity, using each event's latest verdict."""
        rows = self._query(
            "SELECT e.severity, v.verdict FROM events e JOIN verdicts v ON v.id = ("
            "SELECT id FROM verdicts WHERE event_id=e.id ORDER BY id DESC LIMIT 1)"
        )
        out: dict[str, Any] = {}
        for severity in ("watch", "warning", "critical", "all"):
            chosen = [r for r in rows if severity in ("all", r["severity"])]
            decided = [r for r in chosen if r["verdict"] != "unsure"]
            confirmed = sum(r["verdict"] == "confirmed" for r in decided)
            out[severity] = {
                "reviewed": len(chosen),
                "confirmed": confirmed,
                "false_alarm": len(decided) - confirmed,
                "precision": confirmed / len(decided) if decided else None,
            }
        return out


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    row["contributions"] = json.loads(row["contributions"])
    row["end"] = row.pop("end_step")
    return row
