"""
src/eda/plots.py
================
All visualisations return a Matplotlib ``Figure`` so they are equally usable
from Streamlit (``st.pyplot``) and from notebooks / tests. No global state,
no ``plt.show()``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; safe under Streamlit & CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.eda.stats import numeric_columns

sns.set_theme(style="whitegrid")


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    sns.histplot(series, bins=bins, kde=True, ax=ax, color="#2563eb")
    ax.set_title(f"Distribution — {column}")
    ax.set_xlabel(column)
    fig.tight_layout()
    return fig


def boxplot(df: pd.DataFrame, column: str, by: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    if by and by in df.columns:
        sns.boxplot(data=df, x=by, y=column, ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{column} by {by}")
    else:
        sns.boxplot(y=pd.to_numeric(df[column], errors="coerce"), ax=ax, color="#16a34a")
        ax.set_title(f"Boxplot — {column}")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    cols = numeric_columns(df)
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(cols) < 2:
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center")
        return fig
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def time_series(df: pd.DataFrame, value: str = "Delay") -> plt.Figure:
    """Plot a numeric metric over time if a Date column exists."""
    fig, ax = plt.subplots(figsize=(8, 4))
    if "Date" not in df.columns or value not in df.columns:
        ax.text(0.5, 0.5, "No Date/metric available for time series", ha="center")
        return fig
    ts = (
        df.dropna(subset=["Date"])
        .assign(**{value: lambda d: pd.to_numeric(d[value], errors="coerce")})
        .groupby(pd.Grouper(key="Date", freq="D"))[value]
        .mean()
        .dropna()
    )
    ax.plot(ts.index, ts.values, marker="o", color="#dc2626")
    ax.set_title(f"{value} over time (daily mean)")
    ax.set_ylabel(value)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def quadrant_chart(scored: pd.DataFrame) -> plt.Figure:
    """Scatter shipments on a probability (x) / impact (y) risk matrix.

    Expects columns: ``probability``, ``impact``, ``quadrant``.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    palette = {
        "Improve": "#dc2626", "Monitor": "#f59e0b",
        "Tolerate": "#16a34a", "Operate": "#2563eb",
    }
    for q, colour in palette.items():
        sub = scored[scored["quadrant"] == q]
        ax.scatter(sub["probability"], sub["impact"], label=q,
                   color=colour, alpha=0.7, edgecolors="white", s=70)
    ax.axvline(3, color="grey", ls="--", lw=1)
    ax.axhline(3, color="grey", ls="--", lw=1)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Probability  (operational likelihood) →")
    ax.set_ylabel("Impact  (financial + reputational) →")
    ax.set_title("Risk Matrix — Quadrant Classification")
    # Corner labels
    ax.text(4.7, 5.3, "IMPROVE", color="#dc2626", fontweight="bold")
    ax.text(1.0, 5.3, "MONITOR", color="#f59e0b", fontweight="bold")
    ax.text(1.0, 0.7, "OPERATE", color="#2563eb", fontweight="bold")
    ax.text(4.7, 0.7, "TOLERATE", color="#16a34a", fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Quadrant")
    fig.tight_layout()
    return fig
