"""
app.py — Enterprise Risk Intelligence Copilot (Logistics Edition)
=================================================================
Interactive Streamlit dashboard wrapping the full agent pipeline.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from config import SAMPLE_DIR
from src.eda import anomaly, plots, stats
from src.eda.stats import numeric_columns
from src.pipeline import run_pipeline
from src.utils.logging_config import get_logger

log = get_logger(__name__)

st.set_page_config(page_title="Risk Intelligence Copilot",
                   page_icon="🛰️", layout="wide")


# ── caching: heavy pipeline runs are memoised on the raw bytes ─────────────
@st.cache_data(show_spinner=False)
def _cached_pipeline(file_bytes: bytes, name: str, use_llm: bool, prefer_chroma: bool):
    buf = io.BytesIO(file_bytes)
    buf.name = name
    return run_pipeline(buf, use_llm=use_llm, prefer_chroma=prefer_chroma)


def _sidebar():
    st.sidebar.title("🛰️ Risk Copilot")
    st.sidebar.caption("Excel → EDA → NLP → Frameworks → Scoring → Advisory")

    use_llm = st.sidebar.toggle("Use Ollama LLM (if available)", value=True)
    prefer_chroma = st.sidebar.toggle("Use ChromaDB retrieval (if available)", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data source")
    uploaded = st.sidebar.file_uploader("Upload logistics Excel/CSV",
                                        type=["xlsx", "xls", "csv"])

    sample_files = sorted(p.name for p in SAMPLE_DIR.glob("*.xlsx"))
    sample_choice = st.sidebar.selectbox(
        "…or pick a bundled sample", ["(none)"] + sample_files)

    return use_llm, prefer_chroma, uploaded, sample_choice


def _resolve_source(uploaded, sample_choice):
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if sample_choice and sample_choice != "(none)":
        path = SAMPLE_DIR / sample_choice
        return path.read_bytes(), sample_choice
    return None, None


def main():
    use_llm, prefer_chroma, uploaded, sample_choice = _sidebar()
    file_bytes, name = _resolve_source(uploaded, sample_choice)

    st.title("Enterprise Risk Intelligence Copilot")
    st.caption("Logistics Edition — multi-agent advisory for executives")

    if file_bytes is None:
        st.info("⬅️ Upload an Excel/CSV file or select a bundled sample to begin.")
        st.markdown(
            "**Expected columns:** Shipment ID, Route, Date (optional), Delay, "
            "Fuel Cost, Staff Count, Complaints, External Signals (optional). "
            "Column names are auto-mapped — minor naming differences are fine."
        )
        return

    with st.spinner("Running multi-agent pipeline…"):
        res = _cached_pipeline(file_bytes, name, use_llm, prefer_chroma)

    df = res.scoring.df

    # ── headline metrics ──────────────────────────────────────────────────
    p = res.scoring.portfolio
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite risk", f"{p.get('composite', 0):.2f} / 5")
    c2.metric("Financial", f"{p.get('financial', 0):.2f}")
    c3.metric("Operational", f"{p.get('operational', 0):.2f}")
    c4.metric("Reputational", f"{p.get('reputational', 0):.2f}")

    tabs = st.tabs(["📋 Overview", "🔎 EDA", "🚨 Anomalies",
                    "🧠 Diagnosis & Frameworks", "🎯 Risk Matrix", "📝 Advisory"])

    # ── Overview ──────────────────────────────────────────────────────────
    with tabs[0]:
        if not res.validation.ok:
            st.error("Validation errors:\n" +
                     "\n".join(f"- {e}" for e in res.validation.errors))
        for w in res.validation.warnings:
            st.warning(w)
        st.subheader("Quadrant distribution")
        qc = pd.Series(res.scoring.quadrant_counts).rename("shipments")
        st.bar_chart(qc)
        st.subheader("Scored data")
        show_cols = [c for c in ["ShipmentID", "Route", "Delay", "FuelCost",
                                 "StaffCount", "Complaints", "financial_score",
                                 "operational_score", "reputational_score",
                                 "composite_score", "quadrant"] if c in df.columns]
        st.dataframe(df[show_cols].sort_values("composite_score", ascending=False),
                     use_container_width=True, height=380)
        st.download_button("⬇️ Download scored CSV",
                           df.to_csv(index=False).encode(),
                           file_name="scored_shipments.csv", mime="text/csv")

    # ── EDA ───────────────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Descriptive statistics")
        st.dataframe(stats.describe(df), use_container_width=True)
        st.subheader("Per-route summary")
        st.dataframe(stats.route_summary(df), use_container_width=True)

        num_cols = numeric_columns(df)
        col = st.selectbox("Metric to visualise", num_cols,
                           index=num_cols.index("Delay") if "Delay" in num_cols else 0)
        g1, g2 = st.columns(2)
        with g1:
            st.pyplot(plots.histogram(df, col))
        with g2:
            by = "delay_bucket" if "delay_bucket" in df.columns else None
            st.pyplot(plots.boxplot(df, col, by=by))
        g3, g4 = st.columns(2)
        with g3:
            st.pyplot(plots.correlation_heatmap(df))
        with g4:
            st.pyplot(plots.time_series(df, col))

    # ── Anomalies ─────────────────────────────────────────────────────────
    with tabs[2]:
        summ = anomaly.summarise(df)
        a1, a2 = st.columns(2)
        a1.metric("Global anomalies", summ["global_anomalies"])
        a2.metric("Route-relative anomalies", summ["route_anomalies"])
        st.subheader("Global anomalies (robust z > 3)")
        st.dataframe(anomaly.zscore_anomalies(df), use_container_width=True)
        st.subheader("Route-relative anomalies")
        st.dataframe(anomaly.route_anomalies(df), use_container_width=True)

    # ── Diagnosis & Frameworks ────────────────────────────────────────────
    with tabs[3]:
        d = res.diagnosis
        st.subheader("Detected themes")
        if d.theme_counts:
            st.bar_chart(pd.Series(d.theme_counts).rename("count"))
        cc1, cc2 = st.columns(2)
        cc1.markdown("**Stakeholders**")
        cc1.write(", ".join(d.stakeholders) or "—")
        cc2.markdown("**Operational constraints**")
        cc2.write("\n".join(f"- {c}" for c in d.constraints) or "—")
        st.subheader("Sample narratives")
        for nar in d.narratives[:8]:
            st.markdown(f"- {nar}")
        st.subheader("Retrieved frameworks (Agent 2)")
        for f in res.frameworks:
            with st.expander(f"{f.source}  ·  relevance {f.score:.2f}"):
                st.write(f.text)

    # ── Risk Matrix ───────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Probability × Impact quadrant chart")
        st.pyplot(plots.quadrant_chart(df))
        st.caption("Improve = act now · Monitor = watch & plan · "
                   "Tolerate = light controls · Operate = business as usual")

    # ── Advisory ──────────────────────────────────────────────────────────
    with tabs[5]:
        brief = res.advisory
        st.caption(f"Generated by: {brief.generated_by}")
        st.markdown(brief.to_markdown())
        st.download_button("⬇️ Download advisory (Markdown)",
                           brief.to_markdown().encode(),
                           file_name="advisory_brief.md", mime="text/markdown")


if __name__ == "__main__":
    main()
