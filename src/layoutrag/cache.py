"""Content-hash disk cache.

Deliberately a dict on disk, not a subsystem.

It exists because the tool indexes one corpus under several chunking strategies at once.
Without it, four strategies means parsing every PDF four times — and with docling at
seconds per page, that is the difference between a demo that runs while someone watches
and one that doesn't.

Keys are content hashes, so nothing depends on file paths or modification times: the same
bytes parsed with the same parser and the same settings always hit. Changing one stage's
settings misses only that stage.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

T = TypeVar("T")

DEFAULT_CACHE_DIR = Path(".layoutrag_cache")


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def hash_file(path: Path) -> str:
    """Content hash of a file, read in chunks so large PDFs don't land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()[:32]


def hash_params(**params: Any) -> str:
    """Stable hash of a stage's settings.

    Sorted keys and a canonical separator, so the same settings always produce the same
    hash regardless of the order they were passed in.
    """
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:32]


class Cache:
    """Pickle-backed store keyed by (stage, content hash, params hash)."""

    def __init__(self, root: Path | str = DEFAULT_CACHE_DIR, *, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def _path(self, stage: str, content_hash: str, params_hash: str) -> Path:
        # Sharded by the first two hex characters so a corpus of thousands of documents
        # doesn't produce one flat directory.
        return self.root / stage / content_hash[:2] / f"{content_hash}-{params_hash}.pkl"

    def get(self, stage: str, content_hash: str, params_hash: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(stage, content_hash, params_hash)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open("rb") as fh:
                value = pickle.load(fh)
        except (pickle.UnpicklingError, EOFError, AttributeError, ModuleNotFoundError):
            # A half-written or stale-format entry is a miss, not a crash. Recomputing is
            # always correct; refusing to start because of a corrupt cache file is not.
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, stage: str, content_hash: str, params_hash: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(stage, content_hash, params_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file then rename, so an interrupted run can never leave a
        # truncated entry that looks valid.
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(value, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def get_or_compute(
        self,
        stage: str,
        content_hash: str,
        params_hash: str,
        compute: Callable[[], T],
    ) -> T:
        cached = self.get(stage, content_hash, params_hash)
        if cached is not None:
            return cast("T", cached)
        value = compute()
        self.put(stage, content_hash, params_hash, value)
        return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*.pkl"))
