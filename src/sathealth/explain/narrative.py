"""Grounded narrative: plain sentences that only restate computed facts.

Every :class:`Sentence` lists the fact keys it uses. :func:`check_grounding` extracts every
number from a sentence and requires it to appear in the formatted values of those facts, so
a sentence cannot state a number that was not computed. The same check guards any optional
LLM rewording (:mod:`sathealth.explain.llm`).

Facts are pre-formatted strings, so the sentence and the check see exactly the same digits.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
DETECTOR_NAMES = {"lstm": "forecast (LSTM)", "transformer": "reconstruction (Transformer)"}


@dataclass(frozen=True)
class Sentence:
    text: str
    facts: tuple[str, ...]


def numbers(text: str) -> set[str]:
    """Number tokens, with thousands separators removed (1,000 and 1000 are the same)."""
    return {m.replace(",", "") for m in NUMBER.findall(text)}


def check_grounding(sentence: Sentence, facts: Mapping[str, str]) -> list[str]:
    """Numbers in ``sentence`` that none of its referenced facts contain."""
    missing = [key for key in sentence.facts if key not in facts]
    if missing:
        return [f"unknown fact {key!r}" for key in missing]
    allowed: set[str] = set()
    for key in sentence.facts:
        allowed |= numbers(facts[key])
    return sorted(numbers(sentence.text) - allowed)


def _rarity(surprise: float) -> int:
    """10 ** surprise to two significant figures: more digits would be false precision."""
    return int(float(f"{10 ** min(surprise, 12):.2g}"))


def _steps(count: str) -> str:
    return f"{count} step" if count == "1" else f"{count} steps"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def build_facts(explanation: Mapping[str, Any]) -> dict[str, str]:
    """Format every number an explanation may state, once."""
    e = explanation
    event, cf, attr = e["event"], e.get("counterfactual"), e.get("attribution")
    facts = {
        "chan": e["chan_id"],
        "spacecraft": e["spacecraft"],
        "severity": event["severity"],
        "kind": event["kind"],
        "start": str(event["start"]),
        "end": str(event["end"]),
        "duration": str(event["end"] - event["start"] + 1),
        "peak_step": str(event["peak_step"]),
        "peak_surprise": _fmt(event["peak_surprise"], 1),
        "rarity": f"1 in {_rarity(event['peak_surprise']):,}",
        "onset": str(e["onset"]),
        "lead": str(event["start"] - e["onset"]),
    }
    for name, value in event["contributions"].items():
        facts[f"surprise_{name}"] = _fmt(value, 1)
    if cf is not None:
        facts |= {
            "cf_low": _fmt(cf["expected_low"]),
            "cf_high": _fmt(cf["expected_high"]),
            "cf_extreme": _fmt(cf["actual_extreme"]),
            "cf_outside": f"{round(100 * cf['outside_fraction'])}%",
            "cf_horizon": str(cf["horizon"]),
            "cf_band": f"{round(100 * cf['band_level'])}%",
        }
    if attr is not None:
        top = attr["top_features"][0]
        facts |= {
            "attr_feature": top["name"],
            "attr_share": f"{round(100 * top['share'])}%",
            "attr_lag": str(attr["top_lags"][0]),
        }
    return facts


def narrate(explanation: Mapping[str, Any]) -> tuple[list[Sentence], dict[str, str]]:
    """Template sentences for one explained event, plus the facts they are grounded in."""
    f = build_facts(explanation)
    event = explanation["event"]
    s = [
        Sentence(
            f"{f['severity'].capitalize()} alert on channel {f['chan']} ({f['spacecraft']}): "
            f"{f['kind']} anomaly from step {f['start']} to {f['end']} "
            f"({f['duration']} steps).",
            ("severity", "chan", "spacecraft", "kind", "start", "end", "duration"),
        ),
        Sentence(
            f"At its peak, step {f['peak_step']}, the behaviour was rarer than "
            f"{f['rarity']} nominal steps (surprise {f['peak_surprise']}).",
            ("peak_step", "rarity", "peak_surprise"),
        ),
    ]
    lead = int(f["lead"])
    if lead > 0:
        s.append(
            Sentence(
                f"The deviation began at step {f['onset']}, {_steps(f['lead'])} before the "
                "alarm was raised.",
                ("onset", "lead"),
            )
        )
    else:
        s.append(
            Sentence(
                f"The deviation began abruptly at step {f['onset']}, when the alarm was raised.",
                ("onset",),
            )
        )
    contributions = event["contributions"]
    if len(contributions) > 1:
        ranked = sorted(contributions, key=lambda k: -contributions[k])
        parts = [f"{DETECTOR_NAMES.get(k, k)} surprise {f['surprise_' + k]}" for k in ranked]
        agree = all(contributions[k] >= 3 for k in ranked)
        lead_in = "Both detectors flagged it" if agree else "Mainly one detector flagged it"
        s.append(
            Sentence(f"{lead_in}: {', '.join(parts)}.", tuple("surprise_" + k for k in ranked))
        )
    if "cf_low" in f:
        s.append(
            Sentence(
                f"Given the commands actually sent, the value was expected to stay between "
                f"{f['cf_low']} and {f['cf_high']}; it reached {f['cf_extreme']}, outside the "
                f"{f['cf_band']} band for {f['cf_outside']} of the first {f['cf_horizon']} steps.",
                ("cf_low", "cf_high", "cf_extreme", "cf_band", "cf_outside", "cf_horizon"),
            )
        )
    if "attr_feature" in f:
        s.append(
            Sentence(
                f"The forecast error at the peak is driven mostly by {f['attr_feature']} "
                f"({f['attr_share']} of the attribution), most strongly "
                f"{_steps(f['attr_lag'])} before the peak.",
                ("attr_feature", "attr_share", "attr_lag"),
            )
        )
    if event["kind"] == "point":
        s.append(
            Sentence(
                "At least one reading left the training range, so a limit check could also "
                "have seen this.",
                ("kind",),
            )
        )
    else:
        s.append(
            Sentence(
                "Every reading stayed within the training range; only the pattern was "
                "abnormal, which a limit check cannot see.",
                ("kind",),
            )
        )
    return s, f


def ungrounded(sentences: Sequence[Sentence], facts: Mapping[str, str]) -> list[str]:
    """All grounding violations of a narrative, as readable strings (empty when grounded)."""
    return [
        f"{sentence.text!r}: {problem}"
        for sentence in sentences
        for problem in check_grounding(sentence, facts)
    ]
