#!/usr/bin/env bash
# setup.sh — one-shot environment bootstrap.
# Usage:  bash setup.sh
set -euo pipefail

echo "▶ Creating virtual environment (.venv)…"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "▶ Upgrading pip…"
pip install --upgrade pip >/dev/null

echo "▶ Installing Python dependencies…"
pip install -r requirements.txt

echo "▶ Downloading spaCy English model…"
python -m spacy download en_core_web_sm || echo "  (spaCy model optional — pipeline has a rule-based fallback)"

echo "▶ Generating sample datasets…"
python scripts/generate_samples.py

echo "▶ (Optional) Ingesting frameworks into ChromaDB…"
python -m src.knowledge.ingest || echo "  (ChromaDB optional — retrieval has a keyword fallback)"

cat <<'EOF'

✅ Setup complete.

Optional — for full local LLM + semantic retrieval:
  1. Install Ollama:        https://ollama.com
  2. Pull models:           ollama pull nomic-embed-text && ollama pull llama3.1

Run the dashboard:
  source .venv/bin/activate
  streamlit run app.py
EOF
