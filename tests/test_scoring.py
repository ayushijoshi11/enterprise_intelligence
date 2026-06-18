"""Tests for Agent 3 risk scoring + quadrant classification."""
import numpy as np
import pandas as pd

from src.agents.risk_scoring import score
from src.data.preprocessing import preprocess


def _frame(n=40, severe=False):
    rng = np.random.default_rng(0)
    base_delay = 30 if severe else 2
    return pd.DataFrame({
        "ShipmentID": [f"S{i}" for i in range(n)],
        "Route": rng.choice(["X", "Y"], n),
        "Delay": np.clip(rng.normal(base_delay, 5, n), 0, None),
        "FuelCost": rng.normal(5000 if severe else 4000, 500, n),
        "StaffCount": rng.integers(8, 15, n),
        "Complaints": rng.poisson(4 if severe else 0.2, n),
        "ExternalSignals": [""] * n,
    })


def test_scores_in_range():
    res = score(preprocess(_frame()).df)
    for col in ("financial_score", "operational_score", "reputational_score",
                "composite_score"):
        assert res.df[col].between(1, 5).all()


def test_quadrant_values_valid():
    res = score(preprocess(_frame(severe=True)).df)
    assert set(res.df["quadrant"]).issubset(
        {"Improve", "Monitor", "Tolerate", "Operate"})


def test_severe_data_scores_higher():
    calm = score(preprocess(_frame(severe=False)).df).portfolio["composite"]
    bad = score(preprocess(_frame(severe=True)).df).portfolio["composite"]
    assert bad > calm


def test_empty_frame_is_safe():
    res = score(pd.DataFrame())
    assert res.portfolio == {}


def test_quadrant_counts_sum_to_rows():
    df = preprocess(_frame()).df
    res = score(df)
    assert sum(res.quadrant_counts.values()) == len(df)
