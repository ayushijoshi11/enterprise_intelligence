"""
scripts/run_cli.py
==================
Headless pipeline runner — useful for batch jobs, CI, or generating an
advisory without launching the dashboard.

Usage:
    python scripts/run_cli.py data/samples/sample_disruption.xlsx
    python scripts/run_cli.py myfile.xlsx --no-llm --out advisory.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Risk Copilot pipeline.")
    ap.add_argument("input", help="Path to logistics Excel/CSV file")
    ap.add_argument("--no-llm", action="store_true", help="Disable Ollama LLM")
    ap.add_argument("--no-chroma", action="store_true", help="Disable ChromaDB")
    ap.add_argument("--out", help="Write advisory markdown to this path")
    args = ap.parse_args()

    res = run_pipeline(args.input, use_llm=not args.no_llm,
                       prefer_chroma=not args.no_chroma)

    print("\n=== PORTFOLIO ===", res.scoring.portfolio)
    print("=== QUADRANTS ===", res.scoring.quadrant_counts)
    md = res.advisory.to_markdown()
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"\nAdvisory written to {args.out}")
    else:
        print("\n" + md)


if __name__ == "__main__":
    main()
