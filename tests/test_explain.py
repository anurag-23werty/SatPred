import json

import numpy as np
import pytest

from sathealth.config import Config
from sathealth.pipeline.evaluate import evaluation_dir_for
from sathealth.pipeline.explain import (
    explain_run,
    feature_name,
    fitted_lstm,
    model_path,
    to_markdown,
)
from sathealth.pipeline.prepare import load_prepared, run_dir_for


def test_rebuilt_model_is_verified_and_cached(explain_config: Config) -> None:
    channel = load_prepared(run_dir_for(explain_config), "A-1")
    det = fitted_lstm(explain_config, channel)
    assert model_path(explain_config, "A-1").is_file()
    again = fitted_lstm(explain_config, channel)  # from cache
    np.testing.assert_array_equal(det.score(channel.test), again.score(channel.test))


def test_a_model_that_differs_from_the_evaluated_one_is_refused(explain_config: Config) -> None:
    saved = evaluation_dir_for(explain_config, "lstm") / "scores" / "M-1.npz"
    with np.load(saved) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["val"] = arrays["val"] + 1.0  # pretend the evaluated model was different
    np.savez_compressed(saved, **arrays)
    with pytest.raises(RuntimeError, match="differs from the evaluated one"):
        fitted_lstm(explain_config, load_prepared(run_dir_for(explain_config), "M-1"))


def test_explain_run_writes_grounded_explanations(explain_config: Config) -> None:
    out = explain_run(explain_config)
    index = json.loads((out / "index.json").read_text())
    assert index, "the injected spike in A-1 raises at least one alarm"
    assert all(row["grounded"] for row in index)
    first = index[0]
    e = json.loads((out / first["chan_id"] / f"event-{first['event']}.json").read_text())
    assert e["onset"] <= e["event"]["start"]
    assert e["grounding_violations"] == []
    assert {"counterfactual", "attribution", "facts", "sentences"} <= e.keys()
    assert e["attribution"]["completeness_gap"] < 0.1
    assert to_markdown(e).startswith("### ")
    with np.load(out / first["chan_id"] / f"event-{first['event']}.npz") as npz:
        assert npz["expected"].shape == npz["lower"].shape
    every = [json.loads(path.read_text()) for path in out.glob("*/event-*.json")]
    assert all(e["grounding_violations"] == [] for e in every)
    only = explain_run(explain_config, channels=["C-1"])
    assert all(r["chan_id"] == "C-1" for r in json.loads((only / "index.json").read_text()))


def test_explain_run_needs_a_decision_run(explain_config: Config) -> None:
    missing = explain_config.model_copy(
        update={"decision": explain_config.decision.model_copy(update={"name": "nope"})}
    )
    with pytest.raises(FileNotFoundError, match="sathealth decide"):
        explain_run(missing)


def test_llm_brief_is_attached_only_when_grounded(
    explain_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    class _Fake:
        def __init__(self) -> None:
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kwargs: object) -> object:
            text = "Alert raised; duration 9999 steps."  # 9999 is invented
            return SimpleNamespace(
                stop_reason="end_turn",
                model="claude-opus-5-5",
                content=[SimpleNamespace(type="text", text=text)],
            )

    monkeypatch.setattr("sathealth.pipeline.explain.llm_default_client", _Fake)
    cfg = explain_config.model_copy(
        update={"explain": explain_config.explain.model_copy(update={"llm": True})}
    )
    out = explain_run(cfg)
    explained = [json.loads(p.read_text()) for p in out.glob("*/event-*.json")]
    assert explained
    for e in explained:
        assert e["llm_brief"]["grounded"] is False
        assert e["llm_brief"]["text"] is None, "an ungrounded brief is never shown"
        assert e["llm_brief"]["invented_numbers"] == ["9999"]


def test_feature_names() -> None:
    assert feature_name(0, (0,)) == "the channel's own recent values"
    assert feature_name(7, (0,)) == "command 7"
