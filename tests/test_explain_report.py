import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sathealth.config import Config
from sathealth.pipeline.explain import explain_run
from sathealth.pipeline.explain_eval import (
    _clean_starts,
    evaluate_grounding,
    evaluate_onset,
    evaluation_channels,
    wilson,
)
from sathealth.pipeline.explain_report import render_event_figure, write_report


def test_evaluation_helpers(explain_config: Config) -> None:
    assert evaluation_channels(explain_config) == ["A-1", "M-1"]
    onset = evaluate_onset(explain_config)
    assert onset["n"] == len(onset["rows"])
    labels = np.zeros(20, dtype=bool)
    labels[10] = True
    starts = _clean_starts(labels, before=3, after=2)
    assert (7 not in starts and 8 in starts.tolist() is False) or 13 not in starts
    assert set(starts.tolist()) == {3, 4, 5, 6, 7, 14, 15, 16, 17}
    lo, hi = wilson(8, 10)
    assert lo < 0.8 < hi
    assert wilson(0, 0) == (0.0, 1.0)


def test_grounding_evaluation_counts_every_sentence(explain_config: Config) -> None:
    out = explain_run(explain_config)
    result = evaluate_grounding(out)
    assert result["violations"] == 0
    assert result["sentences"] >= result["narratives"] > 0
    assert result["grounded_share"] == 1.0


def test_figure_is_valid_svg() -> None:
    steps = np.arange(100, 160, dtype=np.float64)
    cf = np.arange(130, 160, dtype=np.float64)
    svg = render_event_figure(
        steps,
        np.sin(steps / 5),
        cf,
        np.zeros(30),
        np.full(30, -0.5),
        np.full(30, 0.5),
        onset=130,
        alarm=(135, 150),
        labels=[(132, 155)],
        title="Channel X-1 (SMAP): critical contextual anomaly",
    )
    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert "onset" in svg
    assert "labelled start" in svg


def test_write_report_end_to_end(explain_config: Config, tmp_path: Path) -> None:
    out = write_report(explain_config, tmp_path / "docs" / "EXPLAINABILITY.md")
    text = out.read_text()
    assert "## Hypotheses" in text
    assert "| H4 |" in text
    assert "E2: attribution faithfulness" in text
    (evaluation,) = explain_config.paths.explanations_dir.glob("*/evaluation.json")
    results = json.loads(evaluation.read_text())
    assert results["grounding"]["violations"] == 0


def test_report_has_no_trailing_whitespace(explain_config: Config, tmp_path: Path) -> None:
    text = write_report(explain_config, tmp_path / "docs" / "E.md").read_text()
    assert all(line == line.rstrip() for line in text.splitlines())
