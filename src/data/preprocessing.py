"""
src/data/preprocessing.py
=========================
Turn a raw canonical DataFrame into a clean, feature-rich, analysis-ready
table. Pure functions + one orchestrator (``preprocess``) so each step is
unit-testable in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import LOGICAL_RANGES
from src.utils.logging_config import get_logger

log = get_logger(__name__)

_NUMERIC = ["Delay", "FuelCost", "StaffCount", "Complaints"]


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    report: dict[str, object] = field(default_factory=dict)


# ── individual steps ──────────────────────────────────────────────────────
def drop_empty_and_dupes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    if "ShipmentID" in df.columns:
        df = df.drop_duplicates(subset="ShipmentID", keep="first")
    return df.reset_index(drop=True)


def impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Median-impute numeric fields; fill categorical/text with sensible defaults."""
    df = df.copy()
    filled: dict[str, int] = {}
    for col in _NUMERIC:
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                median = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
                filled[col] = n
    for col in ("Route", "ExternalSignals"):
        if col in df.columns:
            n = int(df[col].isna().sum())
            if n:
                df[col] = df[col].fillna("Unknown")
                filled[col] = n
    return df, filled


def clip_to_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Clamp obviously invalid numeric values to the configured bounds."""
    df = df.copy()
    for col, (lo, hi) in LOGICAL_RANGES.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)
    return df


def detect_outliers(df: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    """Flag per-column outliers via robust z-score (median / MAD).

    Adds boolean ``<col>_outlier`` columns plus a row-level ``is_outlier``.
    Robust statistics avoid the masking effect a single extreme has on mean/std.
    """
    df = df.copy()
    out_cols: list[str] = []
    for col in _NUMERIC:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        med = series.median()
        mad = (series - med).abs().median()
        scale = mad * 1.4826 if mad else series.std(ddof=0)
        if not scale or np.isnan(scale):
            robust_z = pd.Series(0.0, index=series.index)
        else:
            robust_z = (series - med).abs() / scale
        flag = robust_z > z
        df[f"{col}_outlier"] = flag.fillna(False)
        out_cols.append(f"{col}_outlier")
    df["is_outlier"] = df[out_cols].any(axis=1) if out_cols else False
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive analytically useful columns.

    * cost_per_staff      – fuel efficiency relative to crew size
    * complaints_per_unit – reputational pressure normaliser
    * delay_bucket        – categorical severity band
    * route_freq          – exposure / concentration per lane
    """
    df = df.copy()
    if {"FuelCost", "StaffCount"}.issubset(df.columns):
        staff = pd.to_numeric(df["StaffCount"], errors="coerce").replace(0, np.nan)
        df["cost_per_staff"] = (pd.to_numeric(df["FuelCost"], errors="coerce") / staff).round(2)

    if {"Complaints", "StaffCount"}.issubset(df.columns):
        staff = pd.to_numeric(df["StaffCount"], errors="coerce").replace(0, np.nan)
        df["complaints_per_staff"] = (
            pd.to_numeric(df["Complaints"], errors="coerce") / staff
        ).round(3)

    if "Delay" in df.columns:
        delay = pd.to_numeric(df["Delay"], errors="coerce")
        df["delay_bucket"] = pd.cut(
            delay,
            bins=[-0.1, 2, 8, 24, np.inf],
            labels=["on_time", "minor", "major", "severe"],
        ).astype("string")

    if "Route" in df.columns:
        df["route_freq"] = df["Route"].map(df["Route"].value_counts())

    # Fill any NaNs introduced by division-by-zero guards.
    for c in ("cost_per_staff", "complaints_per_staff"):
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df


# ── orchestrator ──────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> PreprocessResult:
    """Full cleaning pipeline. Returns clean DataFrame + a step-by-step report."""
    report: dict[str, object] = {"rows_in": len(df)}

    df = drop_empty_and_dupes(df)
    report["rows_after_dedupe"] = len(df)

    df, filled = impute_missing(df)
    report["imputed"] = filled

    df = clip_to_ranges(df)
    df = detect_outliers(df)
    report["outlier_rows"] = int(df["is_outlier"].sum())

    df = engineer_features(df)
    report["rows_out"] = len(df)
    report["columns_out"] = list(df.columns)

    log.info("Preprocess: %s", {k: report[k] for k in
             ("rows_in", "rows_out", "outlier_rows")})
    return PreprocessResult(df=df, report=report)
