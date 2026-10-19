from pathlib import Path

import pytest

from sathealth.config import Config, load_config
from sathealth.data.download import Manifest
from sathealth.pipeline.ablation import VARIANTS, render_ablation, run_ablation, variant_config
from sathealth.pipeline.evaluate import evaluate_detector
from sathealth.pipeline.prepare import prepare_dataset


@pytest.fixture
def config(raw_root: Path, raw_manifest: Manifest, tmp_path: Path) -> Config:
    cfg = load_config(
        None,
        overrides=[
            f"paths.raw_dir={raw_root}",
            f"paths.processed_dir={tmp_path / 'processed'}",
            f"paths.runs_dir={tmp_path / 'runs'}",
            f"paths.decisions_dir={tmp_path / 'decisions'}",
            "evaluation.bootstrap=20",
            "decision.inputs=[zscore, limit]",
        ],
    )
    prepare_dataset(cfg, manifest=raw_manifest)
    for name in ("zscore", "limit", "iforest"):
        evaluate_detector(cfg, name)
    return cfg


def test_every_variant_is_a_valid_config(config: Config) -> None:
    names = [v.name for v in VARIANTS]
    assert names[0] == "fused"
    assert len(set(names)) == len(names)
    assert [v.name for v in VARIANTS if v.post_hoc] == ["fused-adaptive"]
    for variant in VARIANTS:
        assert variant_config(config, variant).decision.name == variant.name


def test_ablation_report(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from sathealth.pipeline import ablation

    small = tuple(
        v.__class__(v.name, v.change, {**v.overrides, "inputs": ("zscore", "limit")}, v.post_hoc)
        for v in VARIANTS
        if v.name in {"fused", "fused-max", "fused-adaptive"}
    )
    monkeypatch.setattr(ablation, "VARIANTS", small)
    rows = run_ablation(config)
    assert len(rows) == 3 * 3
    text = render_ablation(rows, config)
    assert "## Pre-registered verdict" in text
    assert "**Verdict:" in text
    assert "fused-adaptive *(post-hoc)*" in text
