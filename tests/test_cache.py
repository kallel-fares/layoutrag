"""Cache behaviour that the multi-strategy demo depends on.

The important property is selective invalidation: changing the chunker's settings must not
throw away the parse. If it does, indexing four strategies costs four parses and the demo
stops being viable.
"""

from __future__ import annotations

from pathlib import Path

from layoutrag.cache import Cache, hash_file, hash_params


def test_roundtrip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("parse", "abc", "params1", {"blocks": [1, 2, 3]})
    assert cache.get("parse", "abc", "params1") == {"blocks": [1, 2, 3]}
    assert cache.hits == 1


def test_miss_returns_none(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    assert cache.get("parse", "nothing-here", "params1") is None
    assert cache.misses == 1
    assert cache.hit_rate == 0.0


def test_different_params_do_not_collide(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("chunk", "doc1", hash_params(size=512), "chunks-512")
    cache.put("chunk", "doc1", hash_params(size=256), "chunks-256")
    assert cache.get("chunk", "doc1", hash_params(size=512)) == "chunks-512"
    assert cache.get("chunk", "doc1", hash_params(size=256)) == "chunks-256"


def test_changing_the_chunker_keeps_the_parse(tmp_path: Path) -> None:
    # The property the whole side-by-side demo rests on.
    cache = Cache(tmp_path)
    parse_params = hash_params(parser="pypdfium2")
    cache.put("parse", "doc1", parse_params, "parsed-document")

    for strategy in ("fixed", "recursive", "semantic", "sentence-window"):
        cache.put("chunk", "doc1", hash_params(strategy=strategy), f"chunks-{strategy}")

    # One parse still serves every strategy.
    assert cache.get("parse", "doc1", parse_params) == "parsed-document"
    assert cache.get("chunk", "doc1", hash_params(strategy="semantic")) == "chunks-semantic"


def test_params_hash_is_order_independent(tmp_path: Path) -> None:
    assert hash_params(a=1, b=2) == hash_params(b=2, a=1)
    assert hash_params(a=1, b=2) != hash_params(a=1, b=3)


def test_file_hash_tracks_content_not_path(tmp_path: Path) -> None:
    one = tmp_path / "one.pdf"
    two = tmp_path / "two.pdf"
    one.write_bytes(b"identical bytes")
    two.write_bytes(b"identical bytes")
    assert hash_file(one) == hash_file(two)

    two.write_bytes(b"different bytes")
    assert hash_file(one) != hash_file(two)


def test_corrupt_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    cache.put("parse", "doc1", "p", "value")

    entry = next(tmp_path.rglob("*.pkl"))
    entry.write_bytes(b"not a pickle")

    assert cache.get("parse", "doc1", "p") is None
    # The bad entry is cleared, so the next run recomputes cleanly.
    assert not entry.exists()


def test_disabled_cache_never_stores(tmp_path: Path) -> None:
    cache = Cache(tmp_path, enabled=False)
    cache.put("parse", "doc1", "p", "value")
    assert cache.get("parse", "doc1", "p") is None
    assert cache.size_bytes() == 0


def test_get_or_compute_only_computes_once(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    calls = 0

    def compute() -> str:
        nonlocal calls
        calls += 1
        return "expensive"

    assert cache.get_or_compute("parse", "doc1", "p", compute) == "expensive"
    assert cache.get_or_compute("parse", "doc1", "p", compute) == "expensive"
    assert calls == 1
