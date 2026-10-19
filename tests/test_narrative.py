from typing import Any

from sathealth.explain.narrative import (
    Sentence,
    build_facts,
    check_grounding,
    narrate,
    numbers,
    ungrounded,
)


def _explanation(**over: Any) -> dict[str, Any]:
    e: dict[str, Any] = {
        "chan_id": "P-1",
        "spacecraft": "SMAP",
        "onset": 1180,
        "event": {
            "start": 1200,
            "end": 1250,
            "peak_step": 1210,
            "peak_surprise": 6.3,
            "severity": "critical",
            "kind": "contextual",
            "contributions": {"lstm": 7.1, "transformer": 5.5},
        },
        "counterfactual": {
            "horizon": 71,
            "band_level": 0.9,
            "expected_low": -0.51,
            "expected_high": 0.42,
            "actual_extreme": 1.873,
            "outside_fraction": 0.64,
        },
        "attribution": {
            "top_features": [{"column": 0, "name": "command 12", "share": 0.71}],
            "top_lags": [3, 1, 2],
        },
    }
    e.update(over)
    return e


def test_every_sentence_of_a_full_narrative_is_grounded() -> None:
    sentences, facts = narrate(_explanation())
    assert ungrounded(sentences, facts) == []
    text = " ".join(s.text for s in sentences)
    assert "Critical alert on channel P-1 (SMAP)" in text
    assert "rarer than 1 in 2,000,000 nominal steps" in text
    assert "20 steps before the alarm" in text
    assert "Both detectors flagged it" in text
    assert "between -0.51 and 0.42; it reached 1.87" in text
    assert "command 12 (71% of the attribution), most strongly 3 steps" in text
    assert "a limit check cannot see" in text


def test_invented_numbers_and_unknown_facts_are_caught() -> None:
    facts = {"onset": "1180"}
    assert check_grounding(Sentence("Began at step 1180.", ("onset",)), facts) == []
    assert check_grounding(Sentence("Began at step 1181.", ("onset",)), facts) == ["1181"]
    assert check_grounding(Sentence("x", ("missing",)), facts) == ["unknown fact 'missing'"]


def test_variants_abrupt_point_single_detector_no_extras() -> None:
    event = {**_explanation()["event"], "kind": "point", "contributions": {"zscore": 4.0}}
    e = _explanation(onset=1200, event=event)
    del e["counterfactual"], e["attribution"]
    sentences, facts = narrate(e)
    text = " ".join(s.text for s in sentences)
    assert "began abruptly at step 1200" in text
    assert "limit check could also" in text
    assert "detector" not in text
    assert ungrounded(sentences, facts) == []
    assert "cf_low" not in build_facts(e)


def test_singular_lag_and_mixed_detectors() -> None:
    event = {**_explanation()["event"], "contributions": {"lstm": 4.0, "transformer": 1.2}}
    e = _explanation(onset=1199, event=event)
    e["attribution"]["top_lags"] = [1]
    text = " ".join(s.text for s in narrate(e)[0])
    assert "1 step before the alarm" in text
    assert "1 step before the peak" in text
    assert "Mainly one detector flagged it" in text


def test_numbers_ignores_thousands_separators() -> None:
    assert numbers("1 in 1,000 and 2.5%") == {"1", "1000", "2.5"}
