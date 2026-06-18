"""Tests for the validation layer."""
import pandas as pd

from src.data.validation import validate


def test_missing_required_is_error():
    df = pd.DataFrame({"ShipmentID": ["A1"], "Route": ["X"]})  # missing Delay etc.
    rep = validate(df)
    assert not rep.ok
    assert any("Missing required" in e for e in rep.errors)


def test_duplicate_ids_warns():
    df = pd.DataFrame({
        "ShipmentID": ["A1", "A1"], "Route": ["X", "X"], "Delay": [1, 2],
        "FuelCost": [1, 2], "StaffCount": [1, 2], "Complaints": [0, 0],
    })
    rep = validate(df)
    assert rep.stats["duplicate_ids"] == 1
    assert any("duplicate" in w.lower() for w in rep.warnings)


def test_range_violation_warns():
    df = pd.DataFrame({
        "ShipmentID": ["A1"], "Route": ["X"], "Delay": [99999],  # out of range
        "FuelCost": [1000], "StaffCount": [5], "Complaints": [0],
    })
    rep = validate(df)
    assert rep.stats["range_violations"].get("Delay", 0) == 1


def test_clean_data_passes():
    df = pd.DataFrame({
        "ShipmentID": ["A1", "A2"], "Route": ["X", "Y"], "Delay": [1.0, 2.0],
        "FuelCost": [4000, 4100], "StaffCount": [10, 11], "Complaints": [0, 1],
    })
    rep = validate(df)
    assert rep.ok
