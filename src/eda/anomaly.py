"""
src/eda/anomaly.py
==================
Lightweight, dependency-free anomaly detection for delays / costs.

Two complementary views:
* ``zscore_anomalies``   – global robust z-score per metric (already flagged in
  preprocessing, re-exposed here for the EDA tab).
* ``route_anomalies``    – contextual: a shipment is anomalous *relative to its
  own route's* typical behaviour (catches lane-specific surprises that a global
  threshold misses).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_METRICS = ["Delay", "FuelCost", "Complaints"]


def zscore_anomalies(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Return rows where any metric exceeds the robust z threshold globally."""
    flags = pd.Series(False, index=df.index)
    for col in _METRICS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        med = s.median()
        mad = (s - med).abs().median()
        scale = mad * 1.4826 if mad else s.std(ddof=0)
        if scale and not np.isnan(scale):
            flags |= ((s - med).abs() / scale) > z
    return df[flags].copy()


def route_anomalies(df: pd.DataFrame, z: float = 2.5) -> pd.DataFrame:
    """Contextual anomalies measured within each route group."""
    if "Route" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    flags = pd.Series(False, index=df.index)
    for col in _METRICS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        grp = s.groupby(df["Route"])
        med = grp.transform("median")
        mad = (s - med).abs().groupby(df["Route"]).transform("median")
        scale = (mad * 1.4826).replace(0, np.nan)
        robust_z = (s - med).abs() / scale
        flags |= robust_z.fillna(0) > z
    out = df[flags].copy()
    out["_reason"] = "route-relative deviation"
    return out


def summarise(df: pd.DataFrame) -> dict[str, int]:
    return {
        "global_anomalies": len(zscore_anomalies(df)),
        "route_anomalies": len(route_anomalies(df)),
    }
