"""Project configuration.

A ten-line .env reader rather than a dependency. Real environment variables always win, so
an exported key is never shadowed by a stale file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = ".env"


def load_env(start: Path | None = None) -> Path | None:
    """Load ``.env`` from ``start`` or the nearest parent that has one.

    Returns the file used, or None. Values already present in the environment are left
    alone — an explicit export should beat a file the author forgot about.
    """
    here = (start or Path.cwd()).resolve()

    for directory in (here, *here.parents):
        candidate = directory / ENV_FILE
        if not candidate.is_file():
            continue

        for line in candidate.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return candidate

    return None


def has_openai_key() -> bool:
    load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))
