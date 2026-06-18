"""Integration tests for agents + the full pipeline (fallback backends)."""
import numpy as np
import pandas as pd

from src.agents.advisory import generate_advisory
from src.agents.framework_retrieval import FrameworkRetriever
from src.agents.nlp_diagnosis import diagnose
from src.agents.risk_scoring import score
from src.data.preprocessing import preprocess
from src.pipeline import run_pipeline


def _frame(n=30):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "ShipmentID": [f"S{i}" for i in range(n)],
        "Route": rng.choice(["Shanghai-LA", "Dubai-Hamburg"], n),
        "Delay": np.clip(rng.normal(12, 8, n), 0, None),
        "FuelCost": rng.normal(5000, 800, n),
        "StaffCount": rng.integers(6, 14, n),
        "Complaints": rng.poisson(2, n),
        "ExternalSignals": rng.choice(["fuel price surge", "port congestion", ""], n),
    })


def test_diagnose_produces_themes_and_tokens():
    diag = diagnose(preprocess(_frame()).df)
    assert diag.narratives
    assert diag.theme_counts
    assert diag.tokens
    assert diag.stakeholders


def test_retriever_keyword_fallback():
    r = FrameworkRetriever(prefer_chroma=False)
    hits = r.retrieve(["fuel", "delay", "complaints"], k=3)
    assert 1 <= len(hits) <= 3
    assert all(h.source for h in hits)


def test_advisory_generation_template():
    df = preprocess(_frame()).df
    diag = diagnose(df)
    scoring = score(df)
    frameworks = FrameworkRetriever(prefer_chroma=False).retrieve(diag.tokens)
    brief = generate_advisory(diag, scoring, frameworks, use_llm=False)
    assert brief.executive_summary
    assert brief.immediate_actions
    assert brief.long_term_strategies
    assert "Executive Risk Advisory" in brief.to_markdown()


def test_full_pipeline_runs():
    res = run_pipeline(_frame(), use_llm=False, prefer_chroma=False)
    assert res.validation.ok
    assert not res.scoring.df.empty
    assert res.advisory.executive_summary
