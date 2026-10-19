"""Structured key=value logging on top of the standard library.

Every log line carries a message plus explicit fields, so logs stay grep-able and can be
parsed by any log shipper without pulling in a logging framework::

    log = get_logger(__name__)
    log.info("file verified", path="train/A-1.npy", bytes=576128)
    # 2026-10-19T10:00:00 INFO  sathealth.x | file verified path=train/A-1.npy bytes=576128
"""

import logging
import sys
from typing import Any

_FIELDS_ATTR = "sathealth_fields"
_configured = False


class _StderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever ``sys.stderr`` is *now*.

    A plain ``StreamHandler(sys.stderr)`` keeps the stream object it was created with. When
    the host later swaps stderr (pytest capture, notebooks, services redirecting output) and
    closes the old one, every log call raises ``ValueError: I/O operation on closed file``.
    """

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, _: Any) -> None:  # set by StreamHandler.__init__; ignored on purpose
        pass


class KeyValueFormatter(logging.Formatter):
    """Render records as ``<time> <LEVEL> <logger> | <message> k=v k=v``."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-5s %(name)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        fields: dict[str, Any] = getattr(record, _FIELDS_ATTR, {})
        if not fields:
            return base
        rendered = " ".join(f"{k}={_render(v)}" for k, v in fields.items())
        return f"{base} {rendered}"


def _render(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


class StructLogger:
    """Thin wrapper that turns keyword arguments into structured fields."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, fields: dict[str, Any]) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, msg, extra={_FIELDS_ATTR: fields}, stacklevel=3)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._log(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log(logging.WARNING, msg, fields)

    def error(self, msg: str, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields)


def configure_logging(level: str | int = "INFO") -> None:
    """Install the key=value handler on the ``sathealth`` logger tree (idempotent)."""
    global _configured
    root = logging.getLogger("sathealth")
    root.setLevel(level)
    if not _configured:
        handler = _StderrHandler()
        handler.setFormatter(KeyValueFormatter())
        root.addHandler(handler)
        root.propagate = False
        _configured = True


def get_logger(name: str) -> StructLogger:
    return StructLogger(logging.getLogger(name))
