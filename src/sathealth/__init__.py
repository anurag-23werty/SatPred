"""sathealth: explainable AI for satellite health monitoring and early anomaly detection."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sathealth")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
