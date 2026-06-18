"""
src/eda/stats.py
================
Descriptive statistics helpers used by the dashboard and the agents.
"""
from __future__ import annotations

import pandas as pd

_NUMERIC = ["Delay", "FuelCost", "StaffCount", "Complaints",
            "cost_per_staff", "complaints_per_staff", "route_freq"]


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in _NUMERIC if c in df.columns]


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / median / variance / std / min / max table for numeric fields."""
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({
        "mean": sub.mean(),
        "median": sub.median(),
        "variance": sub.var(ddof=1),
        "std": sub.std(ddof=1),
        "min": sub.min(),
        "max": sub.max(),
        "missing": sub.isna().sum(),
    })
    return out.round(2)


def route_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-route aggregates – the executive's 'where is the pain' view."""
    if "Route" not in df.columns:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {}
    if "Delay" in df.columns:
        agg["avg_delay"] = ("Delay", "mean")
        agg["max_delay"] = ("Delay", "max")
    if "FuelCost" in df.columns:
        agg["avg_fuel"] = ("FuelCost", "mean")
    if "Complaints" in df.columns:
        agg["total_complaints"] = ("Complaints", "sum")
    if "ShipmentID" in df.columns:
        agg["shipments"] = ("ShipmentID", "count")
    if not agg:
        return pd.DataFrame()
    return df.groupby("Route").agg(**agg).round(2).sort_values(
        by=list(agg)[0], ascending=False
    )
