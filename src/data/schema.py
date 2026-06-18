"""
src/data/schema.py
==================
Schema-aware loading of the logistics Excel input.

Responsibilities
----------------
* Read .xlsx / .csv into a DataFrame.
* Map loosely-named user columns to the canonical schema (case / space
  insensitive, common aliases).
* Coerce dtypes per ``config.SCHEMA``.
"""
from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from config import SCHEMA
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Map of lowercase-no-space aliases -> canonical column name.
_ALIASES: dict[str, str] = {
    "shipmentid": "ShipmentID", "shipment": "ShipmentID", "id": "ShipmentID",
    "route": "Route", "lane": "Route", "corridor": "Route",
    "date": "Date", "shipdate": "Date", "departuredate": "Date",
    "delay": "Delay", "delayhrs": "Delay", "delayhours": "Delay",
    "fuelcost": "FuelCost", "fuel": "FuelCost", "fuelcostusd": "FuelCost",
    "staffcount": "StaffCount", "staff": "StaffCount", "headcount": "StaffCount",
    "complaints": "Complaints", "complaintcount": "Complaints",
    "externalsignals": "ExternalSignals", "signals": "ExternalSignals",
    "notes": "ExternalSignals", "externalsignal": "ExternalSignals",
}


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema using the alias table."""
    rename: dict[str, str] = {}
    for col in df.columns:
        canon = _ALIASES.get(_normalise(col))
        if canon:
            rename[col] = canon
    if rename:
        log.info("Mapped columns: %s", rename)
    return df.rename(columns=rename)


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce each known column to its schema dtype, tolerating bad cells."""
    out = df.copy()
    for col, (dtype, _req, _label) in SCHEMA.items():
        if col not in out.columns:
            continue
        try:
            if dtype == "datetime":
                out[col] = pd.to_datetime(out[col], errors="coerce")
            elif dtype == "float":
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif dtype == "int":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            else:  # string
                out[col] = out[col].astype("string")
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("Could not coerce %s to %s: %s", col, dtype, exc)
    return out


def load_excel(source: str | Path | IO, sheet_name: int | str = 0) -> pd.DataFrame:
    """Load an Excel/CSV source, map columns, and coerce dtypes.

    ``source`` may be a path or a file-like object (e.g. a Streamlit upload).
    """
    name = getattr(source, "name", str(source))
    if str(name).lower().endswith(".csv"):
        df = pd.read_csv(source)
    else:
        df = pd.read_excel(source, sheet_name=sheet_name)
    log.info("Loaded %d rows x %d cols from %s", len(df), df.shape[1], name)

    df = map_columns(df)
    df = coerce_dtypes(df)
    return df
