"""
src/data/validation.py
======================
Structural & logical validation of the canonical DataFrame.

``validate`` returns a ``ValidationReport`` rather than raising, so the UI
can surface issues while still letting the user proceed where sensible.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from config import LOGICAL_RANGES, SCHEMA
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)      # block-worthy
    warnings: list[str] = field(default_factory=list)     # informational
    stats: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"Validation: {'PASS' if self.ok else 'FAIL'}"]
        lines += [f"  ERROR:   {e}" for e in self.errors]
        lines += [f"  WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationReport:
    rep = ValidationReport()

    # 1. Required columns present
    required = [c for c, (_d, req, _l) in SCHEMA.items() if req]
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"Missing required columns: {missing}")

    # 2. Duplicate shipment IDs
    if "ShipmentID" in df.columns:
        dupes = df["ShipmentID"].dropna().duplicated().sum()
        if dupes:
            rep.warnings.append(f"{dupes} duplicate ShipmentID value(s) found.")
        rep.stats["duplicate_ids"] = int(dupes)

    # 3. Fully empty rows
    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        rep.warnings.append(f"{empty_rows} completely empty row(s).")

    # 4. Logical range checks
    range_violations: dict[str, int] = {}
    for col, (lo, hi) in LOGICAL_RANGES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=series.index)
        if lo is not None:
            mask |= series < lo
        if hi is not None:
            mask |= series > hi
        n = int(mask.sum())
        if n:
            range_violations[col] = n
            rep.warnings.append(
                f"{n} value(s) in '{col}' outside logical range [{lo}, {hi}]."
            )
    rep.stats["range_violations"] = range_violations

    # 5. Missing-value summary per required field
    miss_summary = {
        c: int(df[c].isna().sum()) for c in required if c in df.columns
    }
    rep.stats["missing_values"] = miss_summary

    log.info("Validation complete: ok=%s errors=%d warnings=%d",
             rep.ok, len(rep.errors), len(rep.warnings))
    return rep
