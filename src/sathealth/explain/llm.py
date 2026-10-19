"""Optional LLM rewording of a narrative, held to the same grounding check.

The template narrative is already complete and grounded. An LLM can turn it into one
readable paragraph for an operator, but it may also invent things. So the reworded text is
accepted only if every number in it appears in the computed facts; otherwise it is
discarded and the template stands. Never used in any evaluation (ADR 0007).

Requires the optional ``llm`` extra (``uv sync --extra llm``) and Anthropic credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sathealth.explain.narrative import Sentence, numbers

SYSTEM = (
    "You rewrite satellite anomaly alerts for spacecraft operators. Rewrite the sentences "
    "you are given as one short, plain paragraph. Use only the facts provided: do not add "
    "causes, recommendations, comparisons or any number that is not in the facts, and copy "
    "every number exactly as written there. Reply with the paragraph only."
)


@dataclass(frozen=True)
class Brief:
    text: str
    grounded: bool
    invented: tuple[str, ...]  # numbers not found in the facts
    model: str


def default_client() -> Any:
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError as exc:
        raise RuntimeError(
            "explain.llm is on but the Anthropic SDK is not installed; run `uv sync --extra llm`"
        ) from exc
    return anthropic.Anthropic()


def reword(
    sentences: Sequence[Sentence],
    facts: Mapping[str, str],
    *,
    model: str = "claude-opus-5-5",
    client: Any = None,
) -> Brief | None:
    """One grounded paragraph, or ``None`` if the model refused or returned no text."""
    client = client or default_client()
    prompt = json.dumps({"sentences": [s.text for s in sentences], "facts": dict(facts)}, indent=2)
    response = client.beta.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return None
    allowed: set[str] = set()
    for value in facts.values():
        allowed |= numbers(value)
    invented = tuple(sorted(numbers(text) - allowed))
    return Brief(text, not invented, invented, str(response.model))
