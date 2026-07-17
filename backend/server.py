#!/usr/bin/env python3
"""Development entrypoint for the standalone PaperAI service."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.paperai.api import create_app  # noqa: E402
from backend.paperai.config import load_settings  # noqa: E402


def main() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
