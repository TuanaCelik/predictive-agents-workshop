"""``python -m app`` — start the local UI on http://localhost:8000"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.keys import ensure_keys  # noqa: E402

ensure_keys("TABPFN_TOKEN", "ANTHROPIC_API_KEY", optional=("TYPESAFE_API_KEY",))

import uvicorn  # noqa: E402

port = int(os.environ.get("PORT", "8000"))
print(f"\nOpen http://localhost:{port}\n")
uvicorn.run("app.server:app", host="127.0.0.1", port=port, reload=False)
