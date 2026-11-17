"""Decide what kind of question we were asked.

Notebook 02 makes this decision with an LLM. Here we use Jev, TypeSafe's
"System One" decision model: you give it some text and typed questions, and it
returns calibrated probabilities instead of generated text. For a routing
decision that's exactly what we want — fast, cheap, and we get a number we can
act on (or refuse to act on when it's not confident).

Set ``TYPESAFE_API_KEY`` to use Jev. If it isn't set, ``route()`` falls back to
the Claude-based classifier from notebook 02 so the agent still works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TASK_TYPES = {
    "prediction": (
        "The user wants an estimate of something that is NOT in the data: "
        "a future outcome, a probability, a score or a value for specific rows "
        "or customers. Words like predict, likely, will, risk, probability, "
        "forecast, score, expected. Answering requires fitting a model."
    ),
    "lookup": (
        "The user wants a fact that can be computed directly from the data as "
        "it is: counts, averages, shares, filters, group-bys, which rows match. "
        "Words like how many, what is the average, list, show me, which "
        "customers have. Answering only requires a pandas query."
    ),
    "other": (
        "Neither — general chit-chat, a question about how the tools work, or "
        "something unrelated to the dataset."
    ),
}


@dataclass
class Route:
    task_type: str
    probabilities: dict[str, float]
    router: str  # "jev" or "claude"

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.task_type, 0.0)


def _route_with_jev(question: str, model: str | None = None) -> Route:
    from typesafe_sdk import Choice, TypeSafeClient

    with TypeSafeClient(model=model) as client:
        response = client.system_one(
            state={"question": question},
            questions={
                "task_type": Choice(
                    instructions=(
                        "`question` is something a user asked an analytics "
                        "assistant that has access to a table of telecom "
                        "customers (plan, tenure, charges, and whether they "
                        "churned). What kind of task is it?"
                    ),
                    criteria=TASK_TYPES,
                )
            },
        )
    answer = response.choices["task_type"]
    return Route(
        task_type=answer.choice,
        probabilities={k: float(v) for k, v in answer.probabilities.items()},
        router="jev",
    )


def _route_with_claude(question: str) -> Route:
    """Fallback: the LLM classifier from notebook 02 (structured output with an enum)."""
    import json

    import anthropic

    client = anthropic.Anthropic()
    schema = {
        "type": "object",
        "properties": {"task_type": {"type": "string", "enum": list(TASK_TYPES)}},
        "required": ["task_type"],
        "additionalProperties": False,
    }
    system = (
        "Classify what kind of task the user's question is, for an assistant that has a "
        "table of telecom customers and whether each churned. Task types:\n"
        + "\n".join(f"- {k}: {v}" for k, v in TASK_TYPES.items())
    )
    msg = client.messages.create(
        model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": question}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    text = next(b.text for b in msg.content if b.type == "text")
    task_type = json.loads(text)["task_type"]
    # An LLM gives us a label, not a distribution. Record that honestly.
    return Route(task_type=task_type, probabilities={task_type: 1.0}, router="claude")


def route(question: str, router: str = "auto") -> Route:
    """Classify ``question`` as prediction / lookup / other.

    router: "jev", "claude", or "auto" (Jev if TYPESAFE_API_KEY is set).
    """
    if router == "auto":
        router = "jev" if os.environ.get("TYPESAFE_API_KEY") else "claude"
    if router == "jev":
        return _route_with_jev(question)
    if router == "claude":
        return _route_with_claude(question)
    raise ValueError(f"unknown router {router!r}")
