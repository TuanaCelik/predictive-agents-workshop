"""Predictive agent for the AI Dev workshop.

Four small pieces:

- ``data``   — loads the telco churn splits and applies the one fix they need.
- ``router`` — decides whether a question is a *prediction* or a *lookup* (Jev).
- ``tools``  — the TabPFN handoff (via the Prior Labs API) and a pandas lookup tool.
- ``agent``  — wires the three together with Claude doing the talking.
"""

__all__ = ["PredictiveAgent", "answer", "route"]


def __getattr__(name):  # lazy, so `agent.data` imports without the LLM SDKs installed
    if name in ("PredictiveAgent", "answer"):
        from . import agent as _agent

        return getattr(_agent, name)
    if name == "route":
        from .router import route

        return route
    raise AttributeError(name)
