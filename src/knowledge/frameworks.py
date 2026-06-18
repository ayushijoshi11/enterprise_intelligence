"""
src/knowledge/frameworks.py
===========================
Seed corpus of risk-management & supply-chain-resilience knowledge used to
populate the vector store. Each entry is a self-contained, retrievable chunk
with metadata tags that the retriever can match against extracted tokens.

This is intentionally paraphrased/original summary content (no copyrighted
text) so the project ships ready-to-run. Swap in your own licensed material
by adding entries with the same structure.
"""
from __future__ import annotations

# Each doc: id, source, tags (keywords the retriever can hit), text.
FRAMEWORK_DOCS: list[dict] = [
    {
        "id": "iso31000-principles",
        "source": "ISO 31000 (Risk Management — Principles)",
        "tags": ["risk", "governance", "framework", "principles", "general"],
        "text": (
            "ISO 31000 frames risk management as creating and protecting value. "
            "It should be integrated into all organisational activities, be "
            "structured and comprehensive, customised to context, inclusive of "
            "stakeholders, dynamic as risks emerge and change, based on the best "
            "available information, account for human and cultural factors, and "
            "be continually improved. The process is: establish scope and "
            "context, then assess (identify, analyse, evaluate), treat, monitor, "
            "review, record and report."
        ),
    },
    {
        "id": "iso31000-treatment",
        "source": "ISO 31000 (Risk Treatment Options)",
        "tags": ["treatment", "mitigation", "avoid", "transfer", "accept", "control"],
        "text": (
            "Risk treatment selects options to modify risk: avoid the activity, "
            "remove the risk source, change likelihood or consequences, share or "
            "transfer the risk (contracts, insurance, hedging), or retain the "
            "risk by informed decision. Treatments can introduce new risks and "
            "should be prioritised by cost-benefit and residual risk tolerance."
        ),
    },
    {
        "id": "scrm-delay-disruption",
        "source": "Supply Chain Risk Management — Disruption Response",
        "tags": ["delay", "disruption", "lead time", "route", "transit", "operational"],
        "text": (
            "Transit delays propagate downstream as stockouts and penalty costs. "
            "Resilience levers include dynamic rerouting around congested or "
            "high-risk lanes, multi-modal flexibility, safety stock and time "
            "buffers at choke points, and pre-negotiated alternative carriers. "
            "Track lead-time variability, not just mean lead time, because "
            "variance drives buffer requirements and service-level failures."
        ),
    },
    {
        "id": "scrm-fuel-cost",
        "source": "Logistics Cost Management — Fuel Exposure",
        "tags": ["fuel", "cost", "price", "hedging", "financial", "spike"],
        "text": (
            "Fuel is a volatile, large share of transport cost. Manage exposure "
            "with fuel surcharge clauses indexed to a public benchmark, forward "
            "contracts or fuel hedging for predictable volumes, route and load "
            "optimisation to cut consumption, and modal shift to lower-cost modes "
            "where lead time allows. Sudden cost spikes warrant variance analysis "
            "to separate price effects from inefficiency or fraud."
        ),
    },
    {
        "id": "scrm-staffing",
        "source": "Operational Resilience — Workforce & Capacity",
        "tags": ["staff", "shortage", "labour", "capacity", "headcount", "operational"],
        "text": (
            "Staff shortages reduce throughput and increase error and delay "
            "rates. Mitigations include cross-training for surge flexibility, a "
            "vetted contingent-labour pool, capacity planning tied to demand "
            "forecasts, and automation of repetitive handling tasks to lower the "
            "headcount-to-volume ratio. Monitor overtime and complaint rates as "
            "leading indicators of capacity strain."
        ),
    },
    {
        "id": "scrm-reputation",
        "source": "Reputational Risk — Service & Complaints",
        "tags": ["complaints", "reputation", "customer", "service", "reputational"],
        "text": (
            "Rising complaint volumes are a leading indicator of reputational and "
            "churn risk. Establish service-level agreements, root-cause complaint "
            "categorisation, proactive customer communication during disruptions, "
            "and a closed-loop corrective-action process. Reputational damage "
            "often lags the operational event, so early, transparent handling "
            "limits long-tail brand and revenue impact."
        ),
    },
    {
        "id": "resilience-diversification",
        "source": "Supply Chain Resilience — Network Design",
        "tags": ["diversification", "multi-port", "supplier", "concentration",
                 "strategic", "resilience", "route"],
        "text": (
            "Concentration in a single lane, port, or supplier is a structural "
            "vulnerability. Build redundancy through multi-port and multi-supplier "
            "strategies, regionalised or near-shored capacity, and the ability to "
            "flex volume between nodes. Quantify concentration with exposure share "
            "per node and stress-test the network against single-node loss."
        ),
    },
    {
        "id": "external-signals",
        "source": "Horizon Scanning — External Signals",
        "tags": ["external", "signals", "weather", "geopolitical", "strike",
                 "macro", "monitoring"],
        "text": (
            "External signals — weather, port congestion, strikes, geopolitical "
            "events, fuel-market moves — are early warnings. Maintain a horizon-"
            "scanning routine that maps signals to affected lanes and pre-defines "
            "trigger-based playbooks so response is fast and rehearsed rather than "
            "improvised during the disruption."
        ),
    },
    {
        "id": "risk-quadrant",
        "source": "Risk Matrix — Prioritisation Doctrine",
        "tags": ["quadrant", "matrix", "prioritise", "impact", "probability",
                 "improve", "monitor", "tolerate", "operate"],
        "text": (
            "Plot risks by probability against impact to prioritise effort. "
            "High-probability / high-impact risks demand active improvement and "
            "investment; high-impact / low-probability risks need monitoring and "
            "contingency plans; low-impact / high-probability risks are tolerated "
            "with light controls; low-impact / low-probability risks are simply "
            "operated through. Re-plot regularly as treatments shift positions."
        ),
    },
]
