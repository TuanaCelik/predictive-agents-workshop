"""Get API keys into the environment without pasting them into notebook cells.

``ensure_keys()`` checks each key in this order and stops at the first hit:

1. already in the environment (exported in the shell, or set by an earlier cell)
2. in the repo's ``.env`` file
3. otherwise, prompts for it with ``getpass`` (input is hidden) and sets it for
   the rest of the session — nothing is written to disk.

Optional keys are only prompted for if you ask; leave the prompt empty to skip.
"""

from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

KEYS = {
    "TABPFN_TOKEN": "Prior Labs API key (platform.priorlabs.ai → API Keys)",
    "ANTHROPIC_API_KEY": "Claude API key (platform.claude.com)",
    "TYPESAFE_API_KEY": "Jev / TypeSafe API key (console.typesafe.ai) — optional, Enter to skip",
}


def ensure_keys(*names: str, optional: tuple[str, ...] = ()) -> dict[str, str]:
    """Make sure each named key is set, prompting only for the ones that aren't.

    Returns ``{name: "env" | ".env" | "prompt" | "skipped"}`` saying where each
    one came from.
    """
    names = names or tuple(k for k in KEYS if k not in optional)
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed — fine, just skip the file
        load_dotenv = None

    before = {k for k in (*names, *optional) if os.environ.get(k)}
    if load_dotenv:
        load_dotenv(ROOT / ".env")

    status: dict[str, str] = {}
    for name in (*names, *optional):
        if name in before:
            status[name] = "env"
        elif os.environ.get(name):
            status[name] = ".env"
        else:
            value = getpass(f"{KEYS.get(name, name)}: ").strip()
            if value:
                os.environ[name] = value
                status[name] = "prompt"
            elif name in optional:
                status[name] = "skipped"
            else:
                raise RuntimeError(f"{name} is required for this notebook.")

    width = max(len(n) for n in status)
    for name, src in status.items():
        label = {"env": "found in environment", ".env": "loaded from .env", "prompt": "set for this session", "skipped": "skipped (optional)"}[src]
        print(f"{name:<{width}}  {label}")
    return status
