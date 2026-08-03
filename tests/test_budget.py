"""Spend guard.

These tests are the reason the guard exists: every one of them describes a way a bill grows
without anyone noticing until the invoice arrives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layoutrag.budget import BudgetExceeded, SpendGuard


def _guard(tmp_path: Path, per_run: float = 1.0, total: float = 10.0) -> SpendGuard:
    return SpendGuard(tmp_path / "spend.json", max_usd_per_run=per_run, max_usd_total=total)


def test_a_run_under_the_ceiling_is_allowed(tmp_path: Path) -> None:
    _guard(tmp_path).check(0.50)


def test_a_run_over_the_per_run_ceiling_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BudgetExceeded, match="per-run ceiling"):
        _guard(tmp_path).check(5.00)


def test_the_cumulative_ceiling_spans_processes(tmp_path: Path) -> None:
    # The oversight this catches: a job that is individually cheap, run in a loop.
    first = _guard(tmp_path, per_run=1.0, total=2.0)
    first.record(0.90, 1000)
    first.commit()

    second = _guard(tmp_path, per_run=1.0, total=2.0)
    second.record(0.90, 1000)
    second.commit()

    third = _guard(tmp_path, per_run=1.0, total=2.0)
    with pytest.raises(BudgetExceeded, match="total ceiling"):
        third.check(0.90)


def test_actual_spend_stops_a_run_that_drifts_past_the_ceiling(tmp_path: Path) -> None:
    # The oversight this catches: an estimate that was too optimistic.
    guard = _guard(tmp_path, per_run=1.0)
    guard.check(0.10)
    guard.record(0.60, 1000)
    with pytest.raises(BudgetExceeded, match="Stopped mid-run"):
        guard.record(0.60, 1000)


def test_an_unreadable_ledger_refuses_rather_than_resetting(tmp_path: Path) -> None:
    # The nastiest failure: a corrupt ledger read as "nothing spent yet" would silently
    # remove the cumulative ceiling, which is the protection that actually matters.
    ledger = tmp_path / "spend.json"
    ledger.write_text("not json")
    with pytest.raises(BudgetExceeded, match="unreadable"):
        _guard(tmp_path).check(0.01)


def test_commit_accumulates_and_resets_the_run(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    guard.record(0.25, 500)
    total = guard.commit()
    assert total.total_usd == pytest.approx(0.25)
    assert total.total_tokens == 500
    assert total.runs == 1
    assert guard.run_usd == 0.0

    guard.record(0.25, 500)
    total = guard.commit()
    assert total.total_usd == pytest.approx(0.50)
    assert total.runs == 2


def test_ledger_is_human_readable(tmp_path: Path) -> None:
    guard = _guard(tmp_path)
    guard.record(0.25, 500)
    guard.commit()
    data = json.loads((tmp_path / "spend.json").read_text())
    assert data["total_usd"] == pytest.approx(0.25)
    assert "updated" in data


def test_ceilings_can_be_raised_by_environment(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("LAYOUTRAG_MAX_USD_PER_RUN", "50")
    monkeypatch.setenv("LAYOUTRAG_MAX_USD_TOTAL", "100")
    SpendGuard(tmp_path / "spend.json").check(40.0)


def test_defaults_are_restrictive(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("LAYOUTRAG_MAX_USD_PER_RUN", raising=False)
    monkeypatch.delenv("LAYOUTRAG_MAX_USD_TOTAL", raising=False)
    guard = SpendGuard(tmp_path / "spend.json")
    assert guard.max_usd_per_run <= 1.0
    assert guard.max_usd_total <= 10.0
