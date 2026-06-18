"""
config.py
=========
Single source of truth for paths, schema, scoring weights and model names.
Every module imports from here so behaviour is tuned in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
CHROMA_DIR = ROOT_DIR / ".chroma"          # persisted vector store
LOG_DIR = ROOT_DIR / "logs"

for _d in (DATA_DIR, SAMPLE_DIR, CHROMA_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Canonical input schema ───────────────────────────────────────────────
# column_name -> (dtype, required, human_label)
SCHEMA: dict[str, tuple[str, bool, str]] = {
    "ShipmentID":     ("string",  True,  "Shipment ID"),
    "Route":          ("string",  True,  "Route"),
    "Date":           ("datetime", False, "Date"),
    "Delay":          ("float",   True,  "Delay (hrs)"),
    "FuelCost":       ("float",   True,  "Fuel Cost (USD)"),
    "StaffCount":     ("int",     True,  "Staff Count"),
    "Complaints":     ("int",     True,  "Complaints"),
    "ExternalSignals": ("string", False, "External Signals"),
}

# Logical ranges used by the validation layer (min, max). None = unbounded.
LOGICAL_RANGES: dict[str, tuple[float | None, float | None]] = {
    "Delay":      (0, 720),       # 0 .. 30 days in hours
    "FuelCost":   (0, 1_000_000),
    "StaffCount": (0, 10_000),
    "Complaints": (0, 100_000),
}

# ── Risk scoring config ──────────────────────────────────────────────────
# Composite weights must sum to 1.0
RISK_WEIGHTS: dict[str, float] = {
    "financial":    0.40,
    "operational":  0.35,
    "reputational": 0.25,
}

# Quadrant thresholds on the 1-5 composite scale.
# Probability axis ~ operational, Impact axis ~ financial+reputational.
QUADRANT_MIDPOINT = 3.0

# ── Model / service names ────────────────────────────────────────────────
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
CHROMA_COLLECTION = "risk_frameworks"
