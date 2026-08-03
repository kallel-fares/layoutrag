"""Spend guard.

A printed estimate is not protection. The classic ways an embedding job produces a shock
bill are all silent: a loop that re-embeds a corpus it already embedded, the wrong model
selected (``-large`` is 6.5x ``-small``), a retry storm, or running the full corpus when a
sample was intended. None of those announce themselves, and all of them are only visible
afterwards on the invoice.

So the ceilings here are enforced in code rather than displayed, they are checked *before*
any request is made, they are re-checked while a job runs so a long job cannot drift past
them, and the running total is persisted to disk so the limit spans runs instead of resetting
every time the process starts.

Defaults are deliberately low. A guard that has to be raised on purpose is doing its job;
one that never fires is decoration.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LEDGER = Path(".layoutrag_spend.json")

# Low on purpose. The whole planned study costs a few dollars, so anything approaching
# these numbers means something is wrong, not that the work grew.
DEFAULT_MAX_USD_PER_RUN = 1.00
DEFAULT_MAX_USD_TOTAL = 10.00


class BudgetExceeded(RuntimeError):
    """Raised instead of spending. Carries the numbers, so the message is actionable."""


@dataclass
class Spend:
    total_usd: float = 0.0
    total_tokens: int = 0
    runs: int = 0


class SpendGuard:
    """Enforces per-run and cumulative ceilings, persisted across processes."""

    def __init__(
        self,
        ledger: Path | str = DEFAULT_LEDGER,
        max_usd_per_run: float | None = None,
        max_usd_total: float | None = None,
    ) -> None:
        self.ledger = Path(ledger)
        # Environment overrides exist so a deliberate large run does not require a code
        # edit, but the default remains restrictive.
        self.max_usd_per_run = (
            max_usd_per_run
            if max_usd_per_run is not None
            else float(os.environ.get("LAYOUTRAG_MAX_USD_PER_RUN", DEFAULT_MAX_USD_PER_RUN))
        )
        self.max_usd_total = (
            max_usd_total
            if max_usd_total is not None
            else float(os.environ.get("LAYOUTRAG_MAX_USD_TOTAL", DEFAULT_MAX_USD_TOTAL))
        )

        self._lock = threading.Lock()
        self.run_usd = 0.0
        self.run_tokens = 0

    def history(self) -> Spend:
        if not self.ledger.exists():
            return Spend()
        try:
            data = json.loads(self.ledger.read_text())
        except (json.JSONDecodeError, OSError):
            # An unreadable ledger must not be read as "nothing spent yet" — that would
            # silently reset the cumulative ceiling, which is the protection that matters.
            raise BudgetExceeded(
                f"Spend ledger at {self.ledger} is unreadable. Refusing to spend without "
                f"knowing the running total. Inspect or delete it deliberately."
            ) from None
        return Spend(
            total_usd=float(data.get("total_usd", 0.0)),
            total_tokens=int(data.get("total_tokens", 0)),
            runs=int(data.get("runs", 0)),
        )

    def check(self, projected_usd: float, *, label: str = "run") -> None:
        """Refuse before spending. Called with an estimate, ahead of any request."""
        if projected_usd > self.max_usd_per_run:
            raise BudgetExceeded(
                f"{label} is projected at ${projected_usd:.4f}, over the per-run ceiling of "
                f"${self.max_usd_per_run:.2f}.\n"
                f"  Raise it deliberately with LAYOUTRAG_MAX_USD_PER_RUN if this is intended."
            )

        prior = self.history().total_usd
        if prior + projected_usd > self.max_usd_total:
            raise BudgetExceeded(
                f"{label} at ${projected_usd:.4f} would take cumulative spend to "
                f"${prior + projected_usd:.4f}, over the total ceiling of "
                f"${self.max_usd_total:.2f} (already spent ${prior:.4f}).\n"
                f"  Raise it deliberately with LAYOUTRAG_MAX_USD_TOTAL if this is intended."
            )

    def record(self, usd: float, tokens: int) -> None:
        """Record actual spend and stop the job if it has drifted past a ceiling.

        Estimates can be wrong. Re-checking as the job proceeds is what stops a long run
        from overshooting after it has already started.
        """
        with self._lock:
            self.run_usd += usd
            self.run_tokens += tokens
            run_usd = self.run_usd

        if run_usd > self.max_usd_per_run:
            raise BudgetExceeded(
                f"Stopped mid-run: actual spend ${run_usd:.4f} passed the per-run ceiling "
                f"of ${self.max_usd_per_run:.2f}."
            )

    def commit(self) -> Spend:
        """Persist this run's spend to the ledger."""
        with self._lock:
            usd, tokens = self.run_usd, self.run_tokens

        prior = self.history()
        updated = Spend(
            total_usd=prior.total_usd + usd,
            total_tokens=prior.total_tokens + tokens,
            runs=prior.runs + 1,
        )
        self.ledger.write_text(
            json.dumps(
                {
                    "total_usd": round(updated.total_usd, 10),
                    "total_tokens": updated.total_tokens,
                    "runs": updated.runs,
                    "last_run_usd": round(usd, 10),
                    "updated": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        with self._lock:
            self.run_usd = 0.0
            self.run_tokens = 0
        return updated
