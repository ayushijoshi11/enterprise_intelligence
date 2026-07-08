"""
src/agents/nlp_diagnosis.py
===========================
Agent 1 — NLP Diagnosis.

Converts the cleaned, scored-ready DataFrame into:
* per-row natural-language *narratives* describing what happened;
* extracted *tokens / themes* (delay, fuel spike, staff shortage, complaints,
  external signals);
* identified *stakeholders* and *operational constraints*.

spaCy is used when available (entity + noun-chunk extraction over the free-text
``ExternalSignals`` column). If the model is not installed, a rule-based
extractor takes over so the agent still produces useful structure.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from config import SPACY_MODEL
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Theme -> trigger keywords used for token tagging on free text + metrics.
_THEME_KEYWORDS: dict[str, list[str]] = {
    "delay": ["delay", "late", "stuck", "congestion", "backlog", "held"],
    "fuel_spike": ["fuel", "diesel", "price", "surcharge", "cost spike"],
    "staff_shortage": ["staff", "shortage", "absence", "sick", "strike", "labour", "labor"],
    "complaints": ["complaint", "dissatisf", "refund", "escalation", "churn"],
    "weather": ["storm", "weather", "flood", "fog", "snow", "typhoon"],
    "geopolitical": ["border", "sanction", "conflict", "geopolitical", "tariff"],
    "port": ["port", "terminal", "berth", "customs", "dock"],
}

_STAKEHOLDERS = ["carrier", "supplier", "customer", "driver", "port authority",
                 "warehouse", "regulator", "insurer", "broker"]


@dataclass
class DiagnosisResult:
    narratives: list[str] = field(default_factory=list)
    theme_counts: dict[str, int] = field(default_factory=dict)
    stakeholders: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)  # for retriever matching


def _load_spacy():
    try:
        import importlib.util
        import importlib.metadata
        import subprocess
        import sys

        if importlib.util.find_spec("spacy") is None:
            raise ImportError("spaCy package not installed")

        proc = subprocess.run(
            [sys.executable, "-c", f"import spacy; spacy.load('{SPACY_MODEL}')"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())

        import spacy
        return spacy.load(SPACY_MODEL)
    except Exception as exc:
        log.info("spaCy model '%s' unavailable (%s); using rule-based NLP.",
                 SPACY_MODEL, type(exc).__name__)
        return None


def _row_narrative(row: pd.Series) -> str:
    """One human-readable sentence per shipment, metric-driven."""
    parts: list[str] = []
    sid = row.get("ShipmentID", "?")
    route = row.get("Route", "unknown route")
    if pd.notna(row.get("Delay")):
        d = float(row["Delay"])
        if d >= 24:
            parts.append(f"severe {d:.0f}h delay")
        elif d >= 8:
            parts.append(f"major {d:.0f}h delay")
        elif d > 2:
            parts.append(f"minor {d:.0f}h delay")
    if row.get("FuelCost_outlier"):
        parts.append("abnormal fuel cost")
    if pd.notna(row.get("StaffCount")) and "StaffCount_outlier" in row and row["StaffCount_outlier"]:
        parts.append("staffing anomaly")
    if pd.notna(row.get("Complaints")) and float(row.get("Complaints", 0)) > 0:
        parts.append(f"{int(row['Complaints'])} complaint(s)")
    sig = row.get("ExternalSignals")
    if isinstance(sig, str) and sig and sig.lower() != "unknown":
        parts.append(f"signal: {sig}")
    body = "; ".join(parts) if parts else "nominal operation"
    return f"Shipment {sid} on {route}: {body}."


def _extract_themes(text: str) -> list[str]:
    text = text.lower()
    hits = []
    for theme, kws in _THEME_KEYWORDS.items():
        if any(k in text for k in kws):
            hits.append(theme)
    return hits


def diagnose(df: pd.DataFrame) -> DiagnosisResult:
    """Run Agent 1 over the cleaned DataFrame."""
    nlp = _load_spacy()
    res = DiagnosisResult()
    theme_counter: Counter[str] = Counter()
    token_set: set[str] = set()
    stakeholder_set: set[str] = set()

    # Aggregate free-text signals for entity extraction.
    all_signals = " ".join(
        str(s) for s in df.get("ExternalSignals", pd.Series(dtype=str)).dropna()
        if str(s).lower() != "unknown"
    )

    for _, row in df.iterrows():
        res.narratives.append(_row_narrative(row))
        blob = f"{row.get('ExternalSignals', '')} {row.get('delay_bucket', '')}"
        # Metric-driven theme tagging
        if pd.notna(row.get("Delay")) and float(row.get("Delay", 0)) > 8:
            theme_counter["delay"] += 1
        if row.get("FuelCost_outlier"):
            theme_counter["fuel_spike"] += 1
        if row.get("StaffCount_outlier"):
            theme_counter["staff_shortage"] += 1
        if pd.notna(row.get("Complaints")) and float(row.get("Complaints", 0)) > 0:
            theme_counter["complaints"] += int(min(float(row["Complaints"]), 1))
        for th in _extract_themes(blob):
            theme_counter[th] += 1

    # spaCy entity / noun-chunk pass over the combined signals text.
    if nlp is not None and all_signals.strip():
        doc = nlp(all_signals[:100_000])
        for ent in doc.ents:
            token_set.add(ent.text.lower())
        for chunk in doc.noun_chunks:
            token_set.add(chunk.root.lemma_.lower())
    # Always also derive tokens from theme keywords actually present.
    for th in theme_counter:
        token_set.update(_THEME_KEYWORDS.get(th, [th]))
        token_set.add(th.replace("_", " "))

    # Stakeholders + constraints from the signal corpus.
    low_signals = all_signals.lower()
    for s in _STAKEHOLDERS:
        if s in low_signals:
            stakeholder_set.add(s)
    # Default stakeholders implied by the data domain.
    stakeholder_set.update({"customer", "carrier", "operations team"})

    constraints: list[str] = []
    if theme_counter.get("staff_shortage"):
        constraints.append("limited workforce capacity")
    if theme_counter.get("fuel_spike"):
        constraints.append("fuel budget exposure")
    if theme_counter.get("delay"):
        constraints.append("lead-time variability / schedule pressure")
    if df.get("route_freq") is not None and "Route" in df.columns:
        top_route = df["Route"].value_counts()
        if len(top_route) and top_route.iloc[0] / len(df) > 0.4:
            constraints.append(f"route concentration on '{top_route.index[0]}'")

    res.theme_counts = dict(theme_counter)
    res.tokens = sorted(token_set)
    res.stakeholders = sorted(stakeholder_set)
    res.constraints = constraints
    log.info("Diagnosis: themes=%s stakeholders=%d tokens=%d",
             res.theme_counts, len(res.stakeholders), len(res.tokens))
    return res
