"""FastAPI application: replay control, alerts, explanations, health, feedback, console.

    uvicorn factory: sathealth.service.app:create_app   (or `sathealth serve`)

The replay runs as a background task that advances ``speed × tick`` steps every tick in a
worker thread and pushes alert updates to every connected console over server-sent events.
OpenAPI docs are served at ``/docs``.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from importlib import resources
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sathealth import __version__
from sathealth.config import Config
from sathealth.logging import get_logger
from sathealth.service.replay import ReplayEngine
from sathealth.service.store import Store

log = get_logger(__name__)

KEEPALIVE_S = 15.0
QUEUE_SIZE = 1000


class Broker:
    """Fan-out of updates to every connected stream. A full queue (a stalled client) drops
    messages instead of blocking the replay; drops are counted."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[tuple[str, dict[str, Any]]]] = set()
        self.dropped = 0

    def subscribe(self) -> asyncio.Queue[tuple[str, dict[str, Any]]]:
        queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(QUEUE_SIZE)
        self._queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[tuple[str, dict[str, Any]]]) -> None:
        self._queues.discard(queue)

    def publish(self, kind: str, data: dict[str, Any]) -> None:
        for queue in self._queues:
            try:
                queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                self.dropped += 1

    @property
    def subscribers(self) -> int:
        return len(self._queues)


class ReplayRequest(BaseModel):
    channels: list[str] | None = None
    speed: float | None = Field(default=None, gt=0, le=100_000)
    run: bool = True


class SpeedRequest(BaseModel):
    speed: float = Field(gt=0, le=100_000)


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=100_000)


class RebaselineRequest(BaseModel):
    operator: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=2000)


class VerdictRequest(BaseModel):
    verdict: Literal["confirmed", "false_alarm", "unsure"]
    operator: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=2000)


async def _runner(app: FastAPI) -> None:
    engine: ReplayEngine = app.state.engine
    broker: Broker = app.state.broker
    tick = app.state.config.service.tick_s
    carry, since_status = 0.0, 0.0
    while True:
        await asyncio.sleep(tick)
        since_status += tick
        if not engine.running or engine.finished:
            carry = 0.0
            continue
        carry += engine.speed * tick
        n, carry = int(carry), carry - int(carry)
        if n:
            for update in await asyncio.to_thread(engine.advance, n):
                broker.publish("alert", update)
        if since_status >= 1.0 or engine.finished:
            since_status = 0.0
            broker.publish("status", engine.status())


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = Store(config.service.db_path)
        app.state.engine = ReplayEngine(config, app.state.store)
        app.state.broker = Broker()
        try:
            await asyncio.to_thread(app.state.engine.reset)
        except FileNotFoundError as exc:  # serve the API, report not ready
            log.error("replay not loaded", reason=str(exc))
            app.state.load_error = str(exc)
        task = asyncio.create_task(_runner(app))
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            app.state.store.close()

    app = FastAPI(
        title="sathealth",
        version=__version__,
        description="Explainable satellite health monitoring: live alerts, explanations, "
        "health and operator feedback.",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.load_error = None

    def engine() -> ReplayEngine:
        if app.state.load_error:
            raise HTTPException(503, app.state.load_error)
        result: ReplayEngine = app.state.engine
        return result

    def store() -> Store:
        result: Store = app.state.store
        return result

    # --- probes and metrics ----------------------------------------------------------------

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz() -> dict[str, Any]:
        if app.state.load_error or app.state.engine.replay_id is None:
            raise HTTPException(503, app.state.load_error or "replay not loaded")
        return {"status": "ready", **app.state.engine.status()}

    @app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
    def metrics() -> str:
        e: ReplayEngine = app.state.engine
        c = e.counters
        health = e.channel_health()
        series = [
            ("replay_steps_total", "counter", "Channel-steps processed", c.steps),
            ("events_opened_total", "counter", "Alerts raised", c.events_opened),
            ("events_closed_total", "counter", "Events finalised", c.events_closed),
            ("explanations_total", "counter", "Explanations computed", c.explanations),
            (
                "explanation_seconds_total",
                "counter",
                "Time spent computing explanations",
                round(c.explanation_seconds, 4),
            ),
            ("stream_subscribers", "gauge", "Connected live streams", app.state.broker.subscribers),
            (
                "stream_dropped_total",
                "counter",
                "Updates dropped for slow clients",
                app.state.broker.dropped,
            ),
            ("replay_step", "gauge", "Current replay step", e.step),
            (
                "health_index_min",
                "gauge",
                "Lowest channel health index",
                min((h["health"] for h in health), default=100.0),
            ),
            (
                "channels_in_alarm",
                "gauge",
                "Channels currently in alarm",
                sum(h["in_alarm"] for h in health),
            ),
        ]
        lines = []
        for name, kind, help_text, value in series:
            lines += [f"# HELP sathealth_{name} {help_text}", f"# TYPE sathealth_{name} {kind}"]
            lines.append(f"sathealth_{name} {value}")
        return "\n".join(lines) + "\n"

    # --- replay ----------------------------------------------------------------------------

    @app.get("/api/status", tags=["replay"])
    def status() -> dict[str, Any]:
        return engine().status()

    @app.post("/api/replay", tags=["replay"])
    def start_replay(body: ReplayRequest) -> dict[str, Any]:
        e = engine()
        try:
            e.reset(body.channels)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        if body.speed is not None:
            e.speed = body.speed
        e.running = body.run
        return e.status()

    @app.post("/api/replay/pause", tags=["replay"])
    def pause() -> dict[str, Any]:
        e = engine()
        e.running = False
        return e.status()

    @app.post("/api/replay/resume", tags=["replay"])
    def resume() -> dict[str, Any]:
        e = engine()
        e.running = not e.finished
        return e.status()

    @app.put("/api/replay/speed", tags=["replay"])
    def set_speed(body: SpeedRequest) -> dict[str, Any]:
        e = engine()
        e.speed = body.speed
        return e.status()

    @app.post("/api/replay/step", tags=["replay"])
    def step(body: StepRequest) -> dict[str, Any]:
        """Advance manually (paused replays, tests and scripted demos)."""
        e = engine()
        updates = e.advance(body.steps)
        for update in updates:
            app.state.broker.publish("alert", update)
        return {"status": e.status(), "updates": updates}

    # --- channels and health ---------------------------------------------------------------

    @app.get("/api/channels", tags=["health"])
    def channels() -> list[dict[str, Any]]:
        return engine().channel_health()

    @app.get("/api/channels/{chan_id}/telemetry", tags=["health"])
    def telemetry(
        chan_id: str, window: Annotated[int, Query(ge=10, le=10_000)] = 600
    ) -> dict[str, Any]:
        try:
            return engine().telemetry(chan_id, window)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/api/drift", tags=["health"])
    def drift() -> list[dict[str, Any]]:
        return engine().drift()

    @app.post("/api/channels/{chan_id}/rebaseline", status_code=201, tags=["health"])
    def rebaseline(chan_id: str, body: RebaselineRequest) -> dict[str, Any]:
        """Operator decision: the channel's recent behaviour is its new normal."""
        try:
            result = engine().rebaseline(chan_id, body.operator, body.note)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        for update in result["updates"]:
            app.state.broker.publish("alert", update)
        app.state.broker.publish("rebaseline", {"chan_id": chan_id, "step": result["step"]})
        return result

    @app.get("/api/channels/{chan_id}/rebaselines", tags=["health"])
    def rebaselines(chan_id: str) -> list[dict[str, Any]]:
        return store().rebaselines(chan_id)

    @app.get("/api/health", tags=["health"])
    def health() -> dict[str, Any]:
        return engine().health()

    # --- events, explanations, verdicts ----------------------------------------------------

    @app.get("/api/events", tags=["events"])
    def events(
        status: Literal["open", "closed"] | None = None,
        severity: Literal["watch", "warning", "critical"] | None = None,
        chan_id: str | None = None,
        all_replays: bool = False,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> list[dict[str, Any]]:
        replay_id = None if all_replays else engine().replay_id
        return store().events(
            replay_id, status=status, severity=severity, chan_id=chan_id, limit=limit
        )

    @app.get("/api/events/{event_id}", tags=["events"])
    def event(event_id: int) -> dict[str, Any]:
        row = store().event(event_id)
        if row is None:
            raise HTTPException(404, f"no event {event_id}")
        return row

    @app.get("/api/events/{event_id}/explanation", tags=["events"])
    def explanation(event_id: int) -> dict[str, Any]:
        try:
            return engine().explain(event_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:  # e.g. a rebuilt model that does not match
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/events/{event_id}/verdict", status_code=201, tags=["feedback"])
    def add_verdict(event_id: int, body: VerdictRequest) -> dict[str, Any]:
        try:
            verdict_id = store().add_verdict(event_id, body.verdict, body.operator, body.note)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        app.state.broker.publish("verdict", {"event_id": event_id, "verdict": body.verdict})
        return {"id": verdict_id, "event_id": event_id, **body.model_dump()}

    @app.get("/api/events/{event_id}/verdicts", tags=["feedback"])
    def verdicts(event_id: int) -> list[dict[str, Any]]:
        return store().verdicts(event_id)

    @app.get("/api/feedback/summary", tags=["feedback"])
    def feedback_summary() -> dict[str, Any]:
        return store().feedback_summary()

    @app.get("/api/feedback/export", tags=["feedback"])
    def feedback_export() -> list[dict[str, Any]]:
        """Every reviewed event with its latest verdict: training data for Phase 7."""
        return [e for e in store().events(None, limit=100_000) if e["verdict"] is not None]

    # --- live stream -----------------------------------------------------------------------

    @app.get("/api/stream", tags=["events"])
    async def stream(request: Request) -> StreamingResponse:
        broker: Broker = app.state.broker
        queue = broker.subscribe()

        async def generate() -> AsyncIterator[str]:
            try:
                yield f"event: status\ndata: {json.dumps(app.state.engine.status())}\n\n"
                while not await request.is_disconnected():
                    try:
                        kind, data = await asyncio.wait_for(queue.get(), KEEPALIVE_S)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"event: {kind}\ndata: {json.dumps(data)}\n\n"
            finally:
                broker.unsubscribe(queue)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- console ---------------------------------------------------------------------------

    static = resources.files("sathealth.service") / "static"
    app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def console() -> str:
        return (static / "index.html").read_text(encoding="utf-8")

    return app
