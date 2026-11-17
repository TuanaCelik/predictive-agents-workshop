"""The predictive agent: Jev decides, TabPFN predicts, Claude talks.

    question ──▶ route()  (Jev: prediction / lookup / other)
                   │
       prediction ─┼─▶ Claude + TabPFN tools   (never trains a model itself)
       lookup ─────┼─▶ Claude + pandas tool
       other ──────┴─▶ Claude, no tools
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable

import anthropic

from .router import Route, route
from .tools import ChurnModel, run_lookup

SYSTEM = """You are an analytics assistant for a telecom company. You have a table of
customers (plan details, tenure, charges) and whether each one churned.

Rules:
- You are the analyst, not the model. For anything that needs a prediction you
  hand off to TabPFN through the tools you are given. Never guess probabilities
  yourself and never suggest training a model by hand.
- Answer in the user's terms, briefly. Include the numbers the tools returned and
  one line on how to read them (the base churn rate in the history is 26.5%).
- If a tool returns an error, say so plainly instead of making something up.
"""

PREDICTION_TOOLS = [
    {
        "name": "score_current_customers",
        "description": (
            "Hand off to TabPFN: score every current (not yet churned) customer "
            "and return the ones most likely to churn, with probabilities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"top_n": {"type": "integer", "description": "How many to return", "default": 10}},
        },
    },
    {
        "name": "score_rows",
        "description": (
            "Hand off to TabPFN: score specific customers the user pasted as CSV "
            "rows (same columns as the dataset, header optional)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"rows_csv": {"type": "string", "description": "The pasted CSV text, verbatim"}},
            "required": ["rows_csv"],
        },
    },
]

LOOKUP_TOOLS = [
    {
        "name": "run_pandas",
        "description": (
            "Run a short pandas snippet against the customer history. A DataFrame "
            "`df` (5,634 labeled customers, column `Churn` is 'Yes'/'No') and `pd` "
            "are in scope. print() what you want to see."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    }
]


@dataclass
class Turn:
    question: str
    route: Route
    answer: str
    tool_calls: list[dict] = field(default_factory=list)


class PredictiveAgent:
    def __init__(self, router: str = "auto", model: str | None = None, min_confidence: float = 0.6):
        self.router = router
        self.model = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.min_confidence = min_confidence
        self.client = anthropic.Anthropic()
        self.churn = ChurnModel()

    # -- tool dispatch ------------------------------------------------------
    def _call_tool(self, name: str, args: dict) -> str:
        try:
            if name == "score_current_customers":
                return json.dumps(self.churn.score_current_customers(int(args.get("top_n", 10))))
            if name == "score_rows":
                return json.dumps(self.churn.score_rows(args["rows_csv"]))
            if name == "run_pandas":
                return run_lookup(args["code"])
            return f"ERROR: unknown tool {name}"
        except Exception as e:  # hand the error to Claude so it can explain or retry
            return f"ERROR: {type(e).__name__}: {e}"

    # -- the loop -------------------------------------------------------------
    def ask(
        self,
        question: str,
        verbose: bool = True,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> Turn:
        """Answer one question.

        ``on_event(kind, payload)`` is called as the work happens, so a UI can
        show it: kinds are ``route``, ``tools``, ``tool_start``, ``tool_end``,
        ``answer``.
        """
        emit = on_event or (lambda kind, payload: None)

        r = route(question, router=self.router)
        emit("route", {"router": r.router, "task_type": r.task_type, "probabilities": r.probabilities, "confidence": r.confidence})
        if verbose:
            probs = ", ".join(f"{k} {v:.0%}" for k, v in sorted(r.probabilities.items(), key=lambda kv: -kv[1]))
            print(f"[router={r.router}] {r.task_type}  ({probs})")

        if r.task_type == "prediction" and r.confidence >= self.min_confidence:
            tools, tool_set = PREDICTION_TOOLS, "prediction"
        elif r.task_type == "lookup" and r.confidence >= self.min_confidence:
            tools, tool_set = LOOKUP_TOOLS, "lookup"
        elif r.task_type == "other":
            tools, tool_set = [], "none"
        else:
            # Not confident enough to commit: give Claude both tool sets and let it ask.
            tools, tool_set = PREDICTION_TOOLS + LOOKUP_TOOLS, "both"
        emit("tools", {"set": tool_set})

        messages: list[dict] = [{"role": "user", "content": question}]
        calls: list[dict] = []
        while True:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM,
                tools=tools,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"[tool] {block.name}({json.dumps(block.input)[:120]})")
                needs_fit = block.name.startswith("score_") and not self.churn.is_fit
                emit("tool_start", {"name": block.name, "input": block.input, "fits_first": needs_fit})
                out = self._call_tool(block.name, block.input)
                calls.append({"tool": block.name, "input": block.input, "output": out[:500]})
                emit("tool_end", {"name": block.name, "output": out[:2000], "fit_seconds": self.churn.fit_seconds if needs_fit else None})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
            messages.append({"role": "user", "content": results})

        text = "".join(b.text for b in resp.content if b.type == "text")
        emit("answer", {"text": text})
        return Turn(question=question, route=r, answer=text, tool_calls=calls)


_default: PredictiveAgent | None = None


def answer(question: str, **kwargs) -> str:
    """One-liner: ``answer("Which customers are most likely to churn?")``."""
    global _default
    if _default is None:
        _default = PredictiveAgent(**kwargs)
    return _default.ask(question).answer
