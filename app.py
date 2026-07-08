"""
app.py — Enterprise Risk Intelligence Copilot (Logistics Edition)
=================================================================
Interactive Streamlit dashboard wrapping the full agent pipeline.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import altair as alt
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

# ── Custom CSS styling for dashboard ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

body {
    color: #0f172a;
    background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    font-family: 'Space Grotesk', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
    color: #0f172a;
    border-right: 1px solid rgba(148, 163, 184, 0.35);
}

div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 18px 22px;
    border-radius: 20px;
    color: #0f172a;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.18);
}

div[data-testid="stTabs"] button {
    background: rgba(203, 213, 225, 0.8);
    color: #0f172a;
    border-radius: 12px;
    margin: 3px 4px;
    font-weight: 600;
    border: 1px solid rgba(148, 163, 184, 0.32);
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #7dd3fc, #a5b4fc);
    color: #0f172a;
    box-shadow: 0 0 18px rgba(59, 130, 246, 0.22);
}

div[data-testid="stExpander"] {
    background-color: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 20px;
}

.css-1d391kg {min-height: 260px;}

.stButton>button {
    background: linear-gradient(135deg, #38bdf8, #a78bfa);
    color: #020617;
    border: none;
    border-radius: 14px;
}

.css-1n76uvr {
    background-color: rgba(255, 255, 255, 0.92) !important;
    border-radius: 18px !important;
}

[data-testid="stSidebar"] .css-1v3fvcr {
    background-color: transparent !important;
}

[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Space Grotesk', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ── caching: heavy pipeline runs are memoised on the raw bytes ─────────────
@st.cache_data(show_spinner=False)
def _cached_pipeline(file_bytes: bytes, name: str, use_llm: bool, prefer_chroma: bool):
    buf = io.BytesIO(file_bytes)
    buf.name = name
    return run_pipeline(buf, use_llm=use_llm, prefer_chroma=prefer_chroma)


def _sidebar():
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Satellite_icon.png/64px-Satellite_icon.png", width=64)
    st.sidebar.title("🛰️ Mission Control")
    st.sidebar.write("Configure the risk analysis experience and control advisory output.")

    use_llm = st.sidebar.checkbox("Enable LLM advisory", value=True)
    prefer_chroma = st.sidebar.checkbox("Use semantic retrieval", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Data Source")
    uploaded = st.sidebar.file_uploader("Upload logistics Excel/CSV", type=["xlsx", "xls", "csv"])

    sample_files = sorted(p.name for p in SAMPLE_DIR.glob("*.xlsx"))
    sample_choice = st.sidebar.selectbox("Pick a sample dataset", ["(none)"] + sample_files)

    st.sidebar.markdown("---")
    threshold = st.sidebar.slider("High-risk threshold", 0.0, 5.0, 4.0, 0.1)
    show_raw = st.sidebar.checkbox("Show raw dataset preview", value=False)

    return use_llm, prefer_chroma, uploaded, sample_choice, threshold, show_raw


def _resolve_source(uploaded, sample_choice):
    if uploaded is not None:
        return uploaded.getvalue(), uploaded.name
    if sample_choice and sample_choice != "(none)":
        path = SAMPLE_DIR / sample_choice
        return path.read_bytes(), sample_choice
    return None, None


def _build_hover_scatter(df: pd.DataFrame) -> alt.Chart | None:
    if "Delay" not in df.columns or "FuelCost" not in df.columns:
        return None

    chart_df = df.copy()
    chart_df["ShipmentID"] = chart_df.get("ShipmentID", chart_df.index.astype(str))
    chart_df["Route"] = chart_df.get("Route", "Unknown")
    chart_df["composite_score"] = chart_df.get("composite_score", 0)
    chart_df["Complaints"] = chart_df.get("Complaints", 0)

    return alt.Chart(chart_df).mark_circle(size=90).encode(
        x=alt.X("Delay", title="Delay (hrs)"),
        y=alt.Y("FuelCost", title="Fuel cost"),
        color=alt.Color("composite_score", scale=alt.Scale(scheme="blues"), title="Composite risk"),
        tooltip=["ShipmentID", "Route", "Delay", "FuelCost", "Complaints", "composite_score"],
        opacity=alt.value(0.85)
    ).interactive()


def _build_problem_chart(df: pd.DataFrame) -> alt.Chart | None:
    if "Delay" not in df.columns or "FuelCost" not in df.columns:
        return None

    global_issues = anomaly.zscore_anomalies(df)
    route_issues = anomaly.route_anomalies(df)
    if global_issues.empty and route_issues.empty:
        return None

    global_issues = global_issues.copy()
    route_issues = route_issues.copy()
    global_issues["_problem_type"] = "global anomaly"
    route_issues["_problem_type"] = "route anomaly"
    problems = pd.concat([global_issues, route_issues], ignore_index=True)
    problems["ShipmentID"] = problems.get("ShipmentID", problems.index.astype(str))
    problems["Route"] = problems.get("Route", "Unknown")
    problems["composite_score"] = problems.get("composite_score", 0)
    problems["Complaints"] = problems.get("Complaints", 0)

    return alt.Chart(problems).mark_circle(size=120).encode(
        x=alt.X("Delay", title="Delay (hrs)"),
        y=alt.Y("FuelCost", title="Fuel cost"),
        color=alt.Color("_problem_type:N", title="Problem type", scale=alt.Scale(domain=["global anomaly", "route anomaly"], range=["#ef4444", "#f59e0b"])),
        tooltip=["ShipmentID", "Route", "Delay", "FuelCost", "Complaints", "composite_score", "_problem_type"]
    ).interactive()


def main():
    use_llm, prefer_chroma, uploaded, sample_choice, threshold, show_raw = _sidebar()
    file_bytes, name = _resolve_source(uploaded, sample_choice)

    st.markdown(
        "<div style='padding:26px; border-radius:28px; background:linear-gradient(135deg, rgba(219,234,254,0.95), rgba(255,255,255,0.95)); margin-bottom:24px;'>"
        "<h1 style='margin-bottom:0.5rem; letter-spacing:0.05em; color:#0f172a;'>Risk Intelligence Copilot</h1>"
        "<p style='color:#334155; margin-top:0;'>A modern analytics dashboard for logistics risk, executive decisioning, and advisory insights.</p>"
        "</div>", unsafe_allow_html=True
    )

    if file_bytes is None:
        st.warning("Choose a dataset from the sidebar to launch the Power BI-style dashboard.")
        st.markdown(
            "<div style='border-radius:24px; background:rgba(255,255,255,0.95); padding:24px; margin-bottom:20px; border:1px solid rgba(148,163,184,0.25);'>"
            "<h2 style='margin-top:0; color:#0f172a;'>Welcome to mission control</h2>"
            "<p style='color:#475569;'>Upload your logistics file or select a sample dataset. Use the navigation tabs to explore the executive overview, diagnostics, anomalies, matrix, and advisory report.</p>"
            "</div>", unsafe_allow_html=True
        )
        return

    with st.spinner("Building the interactive dashboard..."):
        res = _cached_pipeline(file_bytes, name, use_llm, prefer_chroma)

    df = res.scoring.df
    route_count = int(df["Route"].nunique()) if "Route" in df.columns else 0
    high_risk = int((df["composite_score"] >= threshold).sum()) if "composite_score" in df.columns else 0
    avg_delay = float(df["Delay"].mean()) if "Delay" in df.columns else 0.0
    avg_fuel = float(df["FuelCost"].mean()) if "FuelCost" in df.columns else 0.0
    total_complaints = int(df["Complaints"].sum()) if "Complaints" in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(
        "<div style='padding:22px; border-radius:22px; background:rgba(219,234,254,0.95); border:1px solid rgba(148,163,184,0.25);'>"
        "<h4 style='margin:0; color:#0f172a;'>Composite risk</h4>"
        f"<p style='margin:8px 0 0; font-size:2.3rem; font-weight:700; color:#0f172a;'>{res.scoring.portfolio.get('composite', 0):.2f}</p>"
        "</div>", unsafe_allow_html=True)
    k2.markdown(
        "<div style='padding:22px; border-radius:22px; background:rgba(254,215,170,0.95); border:1px solid rgba(148,163,184,0.25);'>"
        "<h4 style='margin:0; color:#0f172a;'>High-risk shipments</h4>"
        f"<p style='margin:8px 0 0; font-size:2.3rem; font-weight:700; color:#0f172a;'>{high_risk}</p>"
        "</div>", unsafe_allow_html=True)
    k3.markdown(
        "<div style='padding:22px; border-radius:22px; background:rgba(221,214,254,0.95); border:1px solid rgba(148,163,184,0.25);'>"
        "<h4 style='margin:0; color:#0f172a;'>Average delay</h4>"
        f"<p style='margin:8px 0 0; font-size:2.3rem; font-weight:700; color:#0f172a;'>{avg_delay:.1f}h</p>"
        "</div>", unsafe_allow_html=True)
    k4.markdown(
        "<div style='padding:22px; border-radius:22px; background:rgba(191,219,254,0.95); border:1px solid rgba(148,163,184,0.25);'>"
        "<h4 style='margin:0; color:#0f172a;'>Routes</h4>"
        f"<p style='margin:8px 0 0; font-size:2.3rem; font-weight:700; color:#0f172a;'>{route_count}</p>"
        "</div>", unsafe_allow_htm l=True)

    tabs = st.tabs([" Overview ", " EDA ", " Anomalies ", " Diagnosis ", " Risk Matrix ", " Advisory "])

    with tabs[0]:
        st.markdown(
            "<div style='border-radius:24px; background:rgba(255,255,255,0.95); padding:24px; margin-bottom:20px; border:1px solid rgba(148,163,184,0.25);'>"
            "<h2 style='margin:0 0 8px; color:#0f172a;'>Executive Overview</h2>"
            "<p style='margin:0; color:#475569;'>A single pane of glass for logistics performance, risk exposure, and priority actions.</p>"
            "</div>", unsafe_allow_html=True)
        left, right = st.columns([2, 1])
        with left:
            st.subheader("Risk quadrant distribution")
            qc = pd.Series(res.scoring.quadrant_counts).rename("shipments")
            st.bar_chart(qc)
            st.markdown(
                "<div style='border-radius:20px; background:rgba(255,255,255,0.95); padding:20px; margin-top:18px; border:1px solid rgba(148,163,184,0.25);'>"
                "<h4 style='margin:0 0 10px; color:#0f172a;'>Risk drivers snapshot</h4>"
                f"<p style='margin:0; color:#475569;'>Avg. fuel cost: <strong>${avg_fuel:.0f}</strong><br>Complaints: <strong>{total_complaints}</strong><br>Threshold: <strong>{threshold:.1f}</strong></p>"
                "</div>", unsafe_allow_html=True)
        with right:
            st.subheader("Top routes by risk")
            if "Route" in df.columns and "composite_score" in df.columns:
                route_scores = df.groupby("Route")["composite_score"].mean().sort_values(ascending=False).head(8)
                st.bar_chart(route_scores)
            else:
                st.info("Route-level risk summary unavailable.")

    with tabs[1]:
        st.markdown(
            "<div style='border-radius:24px; background:rgba(255,255,255,0.92); padding:24px; margin-bottom:20px; border:1px solid rgba(148,163,184,0.25);'>"
            "<h2 style='margin:0 0 8px; color:#0f172a;'>Exploratory Data Analysis</h2>"
            "<p style='margin:0; color:#475569;'>Visualize distributions, correlations, and time-series trends across the dataset.</p>"
            "</div>", unsafe_allow_html=True)

        scatter = _build_hover_scatter(df)
        if scatter is not None:
            st.subheader("Interactive shipment risk scatter")
            st.altair_chart(scatter, use_container_width=True)
            st.caption("Hover over a point to see the exact shipment, route, delay, fuel cost, and risk score.")
        else:
            st.info("Delay and FuelCost data are required for interactive shipment hover charts.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Delay distribution")
            st.pyplot(plots.histogram(df, "Delay"))
        with c2:
            st.subheader("Fuel cost boxplot")
            st.pyplot(plots.boxplot(df, "FuelCost", by="Route"))

        st.subheader("Correlation landscape")
        st.pyplot(plots.correlation_heatmap(df))

        if "Date" in df.columns and "Delay" in df.columns:
            st.subheader("Delay over time")
            st.pyplot(plots.time_series(df, "Delay"))
        else:
            st.info("Date or Delay column missing for time series analysis.")

        if "ShipmentID" in df.columns and "composite_score" in df.columns:
            top_issues = df.sort_values("composite_score", ascending=False).head(12)
            display_cols = [c for c in ["ShipmentID", "Route", "Delay", "FuelCost", "Complaints", "composite_score"] if c in top_issues.columns]
            st.subheader("Top risk shipments")
            st.dataframe(top_issues[display_cols], use_container_width=True)

    with tabs[2]:
        st.markdown(
            "<div style='border-radius:24px; background:rgba(255,255,255,0.92); padding:24px; margin-bottom:20px; border:1px solid rgba(148,163,184,0.25);'>"
            "<h2 style='margin:0 0 8px; color:#0f172a;'>Anomaly Intelligence</h2>"
            "<p style='margin:0; color:#475569;'>Pinpoint outliers and route deviations so you can see exactly which shipment has the problem.</p>"
            "</div>", unsafe_allow_html=True)
        summ = anomaly.summarise(df)
        a1, a2, a3 = st.columns(3)
        a1.metric("Global anomalies", summ["global_anomalies"])
        a2.metric("Route anomalies", summ["route_anomalies"])
        a3.metric("Above threshold", f"{high_risk}")

        problem_chart = _build_problem_chart(df)
        if problem_chart is not None:
            st.subheader("Problem shipment scatter")
            st.altair_chart(problem_chart, use_container_width=True)
            st.caption("Hover over a point to reveal the shipment, route, delay, fuel cost, and reason for the anomaly.")
        else:
            st.info("Anomaly chart needs Delay and FuelCost data.")

        route_tbl = anomaly.route_anomalies(df)
        global_tbl = anomaly.zscore_anomalies(df)
        if not global_tbl.empty:
            st.subheader("Global anomaly shipments")
            st.dataframe(global_tbl.head(20), use_container_width=True)
        if not route_tbl.empty:
            st.subheader("Route-relative anomaly shipments")
            st.dataframe(route_tbl.head(20), use_container_width=True)

    with tabs[3]:
        st.markdown(
            "<div style='border-radius:24px; background:rgba(168,85,247,0.16); padding:24px; margin-bottom:20px;'>"
            "<h2 style='margin:0 0 8px; color:#ffffff;'>Diagnosis & Frameworks</h2>"
            "<p style='margin:0; color:#cbd5e1;'>Understand themes, stakeholders, and constraints behind the current risk posture.</p>"
            "</div>", unsafe_allow_html=True)
        d = res.diagnosis
        if d.theme_counts:
            st.subheader("Theme frequency")
            st.bar_chart(pd.Series(d.theme_counts).rename("count"))
        s1, s2 = st.columns(2)
        s1.markdown(
            "<div style='border-radius:18px; background:rgba(15,23,42,0.88); padding:18px;'>"
            "<h4 style='color:#ffffff;'>Stakeholders</h4>"
            f"<p style='color:#cbd5e1;'>{', '.join(d.stakeholders) if d.stakeholders else 'None detected'}</p>"
            "</div>", unsafe_allow_html=True)
        s2.markdown(
            "<div style='border-radius:18px; background:rgba(15,23,42,0.88); padding:18px;'>"
            "<h4 style='color:#ffffff;'>Constraints</h4>"
            f"<p style='color:#cbd5e1;'>{'<br>'.join(d.constraints) if d.constraints else 'None detected'}</p>"
            "</div>", unsafe_allow_html=True)
        st.subheader("Narrative highlights")
        for narrative in d.narratives[:6]:
            st.markdown(f"- {narrative}")
        st.subheader("Framework recommendations")
        for framework in res.frameworks:
            with st.expander(f"{framework.source} · relevance {framework.score:.2f}"):
                st.write(framework.text)

    with tabs[4]:
        st.markdown(
            "<div style='border-radius:24px; background:rgba(34,197,94,0.14); padding:24px; margin-bottom:20px;'>"
            "<h2 style='margin:0 0 8px; color:#ffffff;'>Risk Matrix</h2>"
            "<p style='margin:0; color:#cbd5e1;'>A visual canvas for probability, impact, and mitigation prioritization.</p>"
            "</div>", unsafe_allow_html=True)
        st.pyplot(plots.quadrant_chart(df))
        st.subheader("Quadrant counts")
        st.bar_chart(pd.Series(res.scoring.quadrant_counts).rename("count"))

    with tabs[5]:
        brief = res.advisory
        st.markdown(
            "<div style='border-radius:24px; background:linear-gradient(135deg, rgba(59,130,246,0.18), rgba(236,72,153,0.18)); padding:24px; margin-bottom:20px;'>"
            "<h2 style='margin:0 0 8px; color:#ffffff;'>Advisory Report</h2>"
            "<p style='margin:0; color:#cbd5e1;'>An executive briefing with findings, actions, and framework context.</p>"
            "</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='border-radius:24px; background:rgba(15,23,42,0.95); padding:24px; color:#e2e8f0;'>"
            f"{brief.to_markdown()}"
            "</div>", unsafe_allow_html=True
        )
        st.download_button("Download advisory (Markdown)", brief.to_markdown().encode(),
                           file_name="advisory_brief.md", mime="text/markdown")


if __name__ == "__main__":
    main()
