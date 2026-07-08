# 🛰️ Enterprise Risk Intelligence Copilot — Logistics Edition

A multi-agent decision-support system that turns raw logistics spreadsheets into
board-ready risk advisories. Drop in an Excel file of shipments and the copilot
runs the full workflow:

```
Excel in → Preprocessing → EDA → NLP Diagnosis → Framework Retrieval
         → Risk Scoring → Executive Advisory out
```

The output is an interactive Streamlit dashboard plus a downloadable advisory
brief (immediate actions + long-term strategy) grounded in risk-management
frameworks (ISO 31000, supply-chain resilience doctrine).

> **Runs out of the box.** spaCy, ChromaDB and Ollama are *optional*. If they
> aren't installed the system automatically falls back to a rule-based NLP
> extractor, a keyword retriever, and a deterministic advisory generator — so
> you always get a complete result. Install them for production-grade semantics.

---
## DEMO LINK

https://enterprise-intelligence.streamlit.app/

## Quick start

```bash
# 1. clone / unzip, then:
bash setup.sh                 # venv + deps + sample data (+ optional ingest)

# 2. run the dashboard
source .venv/bin/activate
streamlit run app.py
```

Open the URL Streamlit prints (default http://localhost:8501), pick a bundled
sample from the sidebar (or upload your own file), and explore the tabs.

### Manual setup (if you prefer)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # optional
python scripts/generate_samples.py             # creates 3 sample xlsx files
streamlit run app.py
```

### Optional: local LLM + semantic retrieval

```bash
# Install Ollama from https://ollama.com then:
ollama pull nomic-embed-text     # embeddings for ChromaDB
ollama pull llama3.1             # advisory narrative generation
python -m src.knowledge.ingest   # build the vector store
```

---

## Input schema

Column names are auto-mapped (case/space-insensitive + common aliases), so
"Shipment ID", "shipment_id" and "ID" all resolve to `ShipmentID`.

| Column            | Type     | Required | Notes                                    |
|-------------------|----------|----------|------------------------------------------|
| ShipmentID        | string   | ✅       | Unique shipment identifier               |
| Route             | string   | ✅       | Lane / corridor, e.g. `Shanghai-LA`      |
| Date              | datetime | optional | Enables time-series view                 |
| Delay             | number   | ✅       | Hours                                     |
| FuelCost          | number   | ✅       | USD                                       |
| StaffCount        | integer  | ✅       | Crew assigned                            |
| Complaints        | integer  | ✅       | Customer complaints linked to shipment   |
| ExternalSignals   | string   | optional | Free text (weather, strikes, congestion) |

Three ready-made samples live in `data/samples/` (baseline, disruption, mixed).

---

## What each tab shows

| Tab | Content |
|-----|---------|
| **Overview** | Validation report, quadrant distribution, scored table, CSV export |
| **EDA** | Descriptive stats, per-route summary, histograms, boxplots, correlation heatmap, time series |
| **Anomalies** | Global (robust z-score) and route-relative anomaly tables |
| **Diagnosis & Frameworks** | Detected themes, stakeholders, constraints, narratives, retrieved frameworks |
| **Risk Matrix** | Probability × Impact quadrant chart |
| **Advisory** | Executive summary, key findings, immediate actions, long-term strategy, framework basis (downloadable) |

---

## Architecture

```
                ┌─────────────┐
  Excel/CSV ──▶ │  Data Layer │  schema map → validate → preprocess
                └──────┬──────┘  (clean, impute, clip, outliers, features)
                       ▼
                ┌─────────────┐
                │   EDA       │  stats · plots · anomaly detection
                └──────┬──────┘
                       ▼
   Agent 1 ──▶  NLP Diagnosis   spaCy / rules → themes, tokens, stakeholders
   Agent 2 ──▶  Framework Retrieval   ChromaDB+Ollama / keyword fallback
   Agent 3 ──▶  Risk Scoring    1–5 financial/operational/reputational + quadrant
   Consultant ▶ Advisory        consolidate → brief (Ollama LLM / template)
                       ▼
                ┌─────────────┐
                │  Streamlit  │  dashboard + downloadable advisory
                └─────────────┘
```

`src/pipeline.py` orchestrates the whole flow and is shared by the dashboard,
the CLI (`scripts/run_cli.py`) and the test suite, so they never drift apart.

---

## Headless / batch usage

```bash
python scripts/run_cli.py data/samples/sample_disruption.xlsx
python scripts/run_cli.py myfile.xlsx --no-llm --out advisory.md
```

## Testing

```bash
pip install pytest
python -m pytest -q          # 19 tests across data, scoring, agents, pipeline
```

## Deployment

* **Local:** `streamlit run app.py`
* **Streamlit Community Cloud:** push to GitHub, point the app at `app.py`.
  Note: Ollama/ChromaDB local services aren't available on Streamlit Cloud, so
  the app will run in fallback mode there (still fully functional). For semantic
  retrieval in the cloud, swap the embedder for a hosted embeddings API.

See [`docs/developer_guide.md`](docs/developer_guide.md) for module-level detail,
the scaling plan, and extension points.

---

## Project layout

```
risk_copilot/
├── app.py                     # Streamlit dashboard
├── config.py                  # schema, weights, model names — tune here
├── requirements.txt
├── setup.sh
├── data/samples/              # generated sample datasets
├── scripts/
│   ├── generate_samples.py
│   └── run_cli.py
├── src/
│   ├── pipeline.py            # orchestrator
│   ├── data/                  # schema · validation · preprocessing
│   ├── eda/                   # stats · plots · anomaly
│   ├── agents/                # nlp_diagnosis · framework_retrieval · risk_scoring · advisory
│   ├── knowledge/             # frameworks corpus · embeddings · chroma ingest
│   └── utils/                 # logging
├── tests/
└── docs/developer_guide.md
```
