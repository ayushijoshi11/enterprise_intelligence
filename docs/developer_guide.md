# Developer Guide

This guide covers internal architecture, extension points, error handling, and a
scaling roadmap. For setup/usage see the top-level [`README.md`](../README.md).

## 1. Design principles

1. **Single orchestrator.** `src/pipeline.run_pipeline()` is the one path from
   raw input to advisory. The dashboard, CLI, and tests all call it, so there is
   no second, drifting implementation of the workflow.
2. **Graceful degradation.** Every external dependency (spaCy, ChromaDB, Ollama)
   is optional. Each integration probes availability at construction time and
   silently swaps in a pure-Python fallback. The system is therefore runnable on
   a laptop with nothing but the core scientific-Python stack.
3. **Pure, testable functions.** Data and EDA steps are side-effect-free
   functions returning new DataFrames/figures. State lives in dataclasses
   (`PreprocessResult`, `DiagnosisResult`, `ScoringResult`, `AdvisoryBrief`,
   `PipelineResult`), not in module globals.
4. **Config in one place.** Schema, scoring weights, thresholds, and model names
   live in `config.py`. Behaviour changes start there.

## 2. Module map

| Module | Responsibility | Key entry points |
|--------|----------------|------------------|
| `config.py` | Schema, ranges, weights, model names | constants |
| `src/data/schema.py` | Load + alias-map columns + coerce dtypes | `load_excel` |
| `src/data/validation.py` | Duplicates, required cols, logical ranges | `validate` → `ValidationReport` |
| `src/data/preprocessing.py` | Clean, impute, clip, outliers, features | `preprocess` → `PreprocessResult` |
| `src/eda/stats.py` | Descriptive stats, route summary | `describe`, `route_summary` |
| `src/eda/plots.py` | Matplotlib figures (headless) | `histogram`, `quadrant_chart`, … |
| `src/eda/anomaly.py` | Global + route-relative anomalies | `zscore_anomalies`, `route_anomalies` |
| `src/agents/nlp_diagnosis.py` | **Agent 1** narratives + tokens | `diagnose` → `DiagnosisResult` |
| `src/agents/framework_retrieval.py` | **Agent 2** RAG / keyword | `FrameworkRetriever.retrieve` |
| `src/agents/risk_scoring.py` | **Agent 3** 1–5 scores + quadrant | `score` → `ScoringResult` |
| `src/agents/advisory.py` | **Consultant** brief generation | `generate_advisory` → `AdvisoryBrief` |
| `src/knowledge/frameworks.py` | Seed corpus (paraphrased, original) | `FRAMEWORK_DOCS` |
| `src/knowledge/embeddings.py` | Ollama / hash-fallback embedder | `get_embedder` |
| `src/knowledge/ingest.py` | Populate ChromaDB | `ingest_frameworks` |
| `src/pipeline.py` | End-to-end orchestrator | `run_pipeline` |
| `src/utils/logging_config.py` | Console + rotating-file logging | `get_logger` |

## 3. The agents in detail

### Agent 1 — NLP Diagnosis
Builds a per-row natural-language narrative from the metrics, then tags
operational themes (delay, fuel_spike, staff_shortage, complaints, weather,
geopolitical, port). When the spaCy model is present it also extracts entities
and noun-chunk lemmas from the free-text `ExternalSignals` column to enrich the
token set used by retrieval. Stakeholders and operational constraints are
inferred from the corpus and from structural signals (e.g. route concentration).

### Agent 2 — Framework Retrieval
`FrameworkRetriever` picks the strongest available backend:
* **ChromaBackend** — embeds the token query (Ollama or hash fallback) and runs
  a vector search over the ingested corpus.
* **KeywordBackend** — scores each framework doc by tag overlap (×2) + body
  keyword hits; always available.

To add frameworks, append entries to `FRAMEWORK_DOCS` (id, source, tags, text)
and re-run `python -m src.knowledge.ingest --force`-style ingestion (or just
delete `.chroma/` and let the app rebuild it). Tags drive both backends, so
choose them to mirror the themes Agent 1 emits.

### Agent 3 — Risk Scoring
Each metric is mapped to a 1–5 band using empirical quantiles (`qcut`) blended
with absolute severity thresholds (e.g. delay buckets), so scores are robust to
units and don't collapse to a single band on a benign dataset. Dimensions:
* `financial` ← FuelCost, cost_per_staff
* `operational` ← Delay (absolute + quantile), staffing anomalies
* `reputational` ← Complaints (absolute + quantile), complaints_per_staff

`composite = Σ weightᵢ · scoreᵢ` with weights in `config.RISK_WEIGHTS`
(must sum to 1.0). Matrix axes: `probability = operational`,
`impact = 0.6·financial + 0.4·reputational`. Quadrants split at
`config.QUADRANT_MIDPOINT`.

### Consultant Agent — Advisory
Consolidates everything into an `AdvisoryBrief` (summary, findings, immediate
actions, long-term strategies, framework basis). Action libraries are keyed by
theme so recommendations track the actual diagnosis. If Ollama is reachable the
executive summary is rewritten by the LLM under a strict "don't invent numbers"
instruction; otherwise the deterministic template summary is used.

## 4. Error handling & logging

* All cross-boundary integrations are wrapped in `try/except` that log at INFO
  (expected absence) or WARNING (unexpected failure) and fall back, never crash.
* `validate()` returns a report rather than raising, so the UI surfaces issues
  while letting the user proceed where reasonable.
* Logs go to console + `logs/app.log` (rotating, 2 MB × 3). Tune in
  `src/utils/logging_config.py`.
* Numeric coercions use `errors="coerce"` + guarded division to avoid
  `#DIV/0`-style blowups on dirty data.

## 5. Testing

`python -m pytest -q` runs 19 tests:
* `test_preprocessing.py` — each cleaning step + full pipeline invariants.
* `test_validation.py` — required cols, duplicates, ranges.
* `test_scoring.py` — score bounds, quadrant validity, severe>calm monotonicity,
  empty-frame safety.
* `test_agents.py` — diagnosis, keyword retrieval, template advisory, full
  pipeline (fallback backends).

Tests deliberately run with `prefer_chroma=False` / `use_llm=False` so they are
deterministic and need no external services in CI.

**Validating scores against history:** to calibrate against real disruptions,
add a labelled dataset under `data/`, write a test asserting that known
high-impact events land in `Improve`/`Monitor`, and tune `RISK_WEIGHTS` /
thresholds until the confusion matrix is acceptable.

## 6. Extension points

* **New metric** → add to `config.SCHEMA` (+ alias in `schema._ALIASES`), include
  it in `preprocessing._NUMERIC`, and reference it in the relevant scoring
  dimension.
* **New framework / industry** → add docs to `FRAMEWORK_DOCS` with tags matching
  your themes; optionally add theme keywords in `nlp_diagnosis._THEME_KEYWORDS`
  and action text in `advisory._IMMEDIATE` / `_LONGTERM`.
* **Different LLM** → change `OLLAMA_LLM_MODEL` / `OLLAMA_EMBED_MODEL` (env vars
  supported) or replace the client in `embeddings.py` / `advisory.py`.

## 7. Scaling plan

**Larger datasets**
* Stream/chunk Excel reads (`pandas` `chunksize` or `openpyxl read_only`).
* Push aggregation to DuckDB/Polars for >1M rows; the EDA + scoring functions are
  column-oriented and port cleanly.
* Cache the pipeline (already done in the dashboard via `st.cache_data`).

**More frameworks**
* The vector store scales to thousands of chunks; switch ChromaDB to a server
  deployment and batch the embedding calls.
* Add chunking + metadata filtering (by industry/region) to keep retrieval
  precise as the corpus grows.

**Industry expansion**
* The data layer, scoring engine, and advisory structure are domain-agnostic.
  To target a new vertical, supply a new schema, a new framework corpus, and
  swap the theme/action libraries — the pipeline and dashboard are unchanged.

**Productionisation**
* Containerise (Dockerfile + the existing `requirements.txt`).
* Run Ollama as a sidecar service; point `OLLAMA_HOST` at it.
* Add authentication in front of Streamlit (reverse proxy / Streamlit auth) for
  executive access, and persist runs to a database for trend tracking.
