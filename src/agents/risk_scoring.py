"""
src/agents/risk_scoring.py
==========================
Agent 3 — Risk Scoring.

Scores each shipment on three 1-5 dimensions and a weighted composite, then
classifies it on the probability/impact risk matrix.

Dimension mapping
-----------------
* financial    ← FuelCost, cost_per_staff (cost burden / volatility)
* operational  ← Delay, delay severity, staffing anomalies
* reputational ← Complaints, complaints_per_staff

Each raw metric is converted to a 1-5 band using empirical quantiles of the
dataset (robust to scale/units) blended with absolute severity thresholds, so a
single benign dataset doesn't inflate everyone to 5 and vice-versa.

Matrix axes
-----------
* probability (x) = operational score (likelihood of recurrence/disruption)
* impact      (y) = weighted blend of financial + reputational
Quadrants (midpoint = config.QUADRANT_MIDPOINT):
    high prob & high impact  -> Improve
    low  prob & high impact  -> Monitor
    high prob & low  impact  -> Tolerate
    low  prob & low  impact  -> Operate
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import QUADRANT_MIDPOINT, RISK_WEIGHTS
from src.utils.logging_config import get_logger

log = get_logger(__name__)


def _band_by_quantile(series: pd.Series) -> pd.Series:
    """Map a numeric series to 1-5 bands using its own quantiles.

    Falls back to all-1s for constant/empty series.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().nunique() <= 1:
        return pd.Series(1, index=s.index, dtype=float)
    # qcut into 5 bins; duplicates='drop' guards against tied edges.
    try:
        bands = pd.qcut(s.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
        return bands.astype(float)
    except Exception:
        # Even spread fallback
        ranks = s.rank(pct=True)
        return np.ceil(ranks * 5).clip(1, 5)


def _absolute_delay_band(delay: pd.Series) -> pd.Series:
    d = pd.to_numeric(delay, errors="coerce").fillna(0)
    bins = [-0.1, 2, 8, 24, 72, np.inf]
    labels = [1, 2, 3, 4, 5]
    return pd.cut(d, bins=bins, labels=labels).astype(float)


def _blend(*series: pd.Series) -> pd.Series:
    """Average several 1-5 bands, ignoring NaNs, clipped to [1,5]."""
    df = pd.concat(series, axis=1)
    return df.mean(axis=1, skipna=True).clip(1, 5)


@dataclass
class ScoringResult:
    df: pd.DataFrame              # original + score columns
    portfolio: dict[str, float]  # aggregate scores
    quadrant_counts: dict[str, int]


def score(df: pd.DataFrame) -> ScoringResult:
    out = df.copy()
    n = len(out)
    if n == 0:
        return ScoringResult(out, {}, {})

    # ── Financial ─────────────────────────────────────────────────────────
    fin_parts = []
    if "FuelCost" in out.columns:
        fin_parts.append(_band_by_quantile(out["FuelCost"]))
    if "cost_per_staff" in out.columns:
        fin_parts.append(_band_by_quantile(out["cost_per_staff"]))
    out["financial_score"] = _blend(*fin_parts) if fin_parts else 1.0

    # ── Operational ───────────────────────────────────────────────────────
    op_parts = []
    if "Delay" in out.columns:
        op_parts.append(_absolute_delay_band(out["Delay"]))
        op_parts.append(_band_by_quantile(out["Delay"]))
    if "StaffCount_outlier" in out.columns:
        op_parts.append(out["StaffCount_outlier"].map({True: 4.0, False: 1.0}))
    out["operational_score"] = _blend(*op_parts) if op_parts else 1.0

    # ── Reputational ──────────────────────────────────────────────────────
    rep_parts = []
    if "Complaints" in out.columns:
        rep_parts.append(_band_by_quantile(out["Complaints"]))
        # absolute floor: any complaint is at least band 2
        comp = pd.to_numeric(out["Complaints"], errors="coerce").fillna(0)
        rep_parts.append(comp.apply(lambda c: 1.0 if c == 0 else min(5.0, 2 + c / 5)))
    if "complaints_per_staff" in out.columns:
        rep_parts.append(_band_by_quantile(out["complaints_per_staff"]))
    out["reputational_score"] = _blend(*rep_parts) if rep_parts else 1.0

    # ── Composite ─────────────────────────────────────────────────────────
    out["composite_score"] = (
        out["financial_score"] * RISK_WEIGHTS["financial"]
        + out["operational_score"] * RISK_WEIGHTS["operational"]
        + out["reputational_score"] * RISK_WEIGHTS["reputational"]
    ).round(2)

    # ── Matrix axes + quadrant ────────────────────────────────────────────
    out["probability"] = out["operational_score"].round(2)
    out["impact"] = (
        out["financial_score"] * 0.6 + out["reputational_score"] * 0.4
    ).round(2)

    mid = QUADRANT_MIDPOINT

    def _quad(r: pd.Series) -> str:
        hi_p = r["probability"] >= mid
        hi_i = r["impact"] >= mid
        if hi_p and hi_i:
            return "Improve"
        if not hi_p and hi_i:
            return "Monitor"
        if hi_p and not hi_i:
            return "Tolerate"
        return "Operate"

    out["quadrant"] = out.apply(_quad, axis=1)

    portfolio = {
        "financial": round(float(out["financial_score"].mean()), 2),
        "operational": round(float(out["operational_score"].mean()), 2),
        "reputational": round(float(out["reputational_score"].mean()), 2),
        "composite": round(float(out["composite_score"].mean()), 2),
    }
    quad_counts = out["quadrant"].value_counts().to_dict()
    log.info("Scoring: portfolio=%s quadrants=%s", portfolio, quad_counts)
    return ScoringResult(df=out, portfolio=portfolio, quadrant_counts=quad_counts)
