"""Local web UI for the predictive agent.

    python -m app            # then open http://localhost:8000

One page, one endpoint. ``POST /ask`` streams the agent's work as server-sent
events (route → tools → tool calls → answer) so the page can light up the
pipeline as it happens. The agent itself is unchanged: ``agent.PredictiveAgent``.
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import PredictiveAgent

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Predictive agent")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_agents: dict[str, PredictiveAgent] = {}
_lock = threading.Lock()


def get_agent(router: str) -> PredictiveAgent:
    """One agent per router choice, so the TabPFN fit is cached across questions."""
    with _lock:
        if router not in _agents:
            a = PredictiveAgent(router=router)
            # share one fitted model between agents
            if _agents:
                a.churn = next(iter(_agents.values())).churn
            _agents[router] = a
        return _agents[router]


class Ask(BaseModel):
    question: str
    router: str = "auto"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/config")
def config() -> dict:
    return {
        "jev": bool(os.environ.get("TYPESAFE_API_KEY")),
        "tabpfn": bool(os.environ.get("TABPFN_TOKEN")),
        "claude": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
    }


@app.post("/ask")
def ask(body: Ask) -> StreamingResponse:
    q: queue.Queue = queue.Queue()

    def on_event(kind: str, payload: dict) -> None:
        q.put((kind, payload))

    def work() -> None:
        try:
            get_agent(body.router).ask(body.question, verbose=False, on_event=on_event)
        except Exception as e:  # surface to the page instead of dying silently
            q.put(("error", {"message": f"{type(e).__name__}: {e}"}))
        finally:
            q.put(("done", {}))

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            kind, payload = q.get()
            yield f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            if kind == "done":
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
