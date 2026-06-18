"""Tests for the data preprocessing layer."""
import numpy as np
import pandas as pd

from src.data.preprocessing import (
    clip_to_ranges, detect_outliers, drop_empty_and_dupes,
    engineer_features, impute_missing, preprocess,
)


def _raw():
    return pd.DataFrame({
        "ShipmentID": ["A1", "A1", "A2", "A3", None],   # dupe + empty-ish
        "Route": ["X", "X", "Y", None, None],
        "Delay": [2.0, 2.0, np.nan, 50.0, np.nan],
        "FuelCost": [4000, 4000, 4200, 99999, np.nan],   # one spike
        "StaffCount": [10, 10, 12, 11, np.nan],
        "Complaints": [0, 0, 1, 8, np.nan],
        "ExternalSignals": ["", "", "fuel surge", None, None],
    })


def test_drop_empty_and_dupes():
    df = drop_empty_and_dupes(_raw())
    assert df["ShipmentID"].is_unique
    assert len(df) == 3  # fully-empty row dropped + one duplicate A1 removed


def test_impute_missing_fills_numeric_median():
    df, filled = impute_missing(_raw())
    assert df["Delay"].isna().sum() == 0
    assert "Delay" in filled and filled["Delay"] >= 1


def test_clip_to_ranges_bounds_values():
    df = clip_to_ranges(_raw())
    assert df["Delay"].max() <= 720
    assert df["FuelCost"].max() <= 1_000_000


def test_detect_outliers_flags_spike():
    df = detect_outliers(impute_missing(_raw())[0])
    assert "is_outlier" in df.columns
    assert df["FuelCost_outlier"].any()


def test_engineer_features_creates_columns():
    df, _ = impute_missing(_raw())
    df = engineer_features(df)
    for c in ("cost_per_staff", "delay_bucket", "route_freq"):
        assert c in df.columns


def test_full_preprocess_pipeline():
    res = preprocess(_raw())
    assert res.report["rows_in"] == 5
    assert res.report["rows_out"] >= 1
    assert "is_outlier" in res.df.columns
    # No NaNs left in core numeric fields
    for c in ("Delay", "FuelCost", "StaffCount", "Complaints"):
        assert res.df[c].isna().sum() == 0
