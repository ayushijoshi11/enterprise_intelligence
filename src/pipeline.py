"""
src/pipeline.py
===============
End-to-end orchestrator: Excel in → preprocessing → EDA → NLP → retrieval →
scoring → advisory out. Used by both the Streamlit app and tests/CLI so the
two never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO

import pandas as pd

from src.agents.advisory import AdvisoryBrief, generate_advisory
from src.agents.framework_retrieval import FrameworkRetriever, Retrieved
from src.agents.nlp_diagnosis import DiagnosisResult, diagnose
from src.agents.risk_scoring import ScoringResult, score
from src.data.preprocessing import PreprocessResult, preprocess
from src.data.schema import load_excel
from src.data.validation import ValidationReport, validate
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class PipelineResult:
    raw: pd.DataFrame
    validation: ValidationReport
    pre: PreprocessResult
    diagnosis: DiagnosisResult
    frameworks: list[Retrieved]
    scoring: ScoringResult
    advisory: AdvisoryBrief


def run_pipeline(source: str | Path | IO | pd.DataFrame,
                 use_llm: bool = True,
                 prefer_chroma: bool = True) -> PipelineResult:
    """Run the full advisory workflow and return every intermediate artefact."""
    # 1. Load
    raw = source if isinstance(source, pd.DataFrame) else load_excel(source)

    # 2. Validate
    report = validate(raw)

    # 3. Preprocess
    pre = preprocess(raw)

    # 4. NLP diagnosis (Agent 1)
    diag = diagnose(pre.df)

    # 5. Framework retrieval (Agent 2)
    retriever = FrameworkRetriever(prefer_chroma=prefer_chroma)
    frameworks = retriever.retrieve(diag.tokens, k=4)

    # 6. Risk scoring (Agent 3)
    scoring = score(pre.df)

    # 7. Advisory (Consultant Agent)
    advisory = generate_advisory(diag, scoring, frameworks, use_llm=use_llm)

    log.info("Pipeline complete: %d rows, %d quadrants, advisory by %s.",
             len(scoring.df), len(scoring.quadrant_counts), advisory.generated_by)
    return PipelineResult(
        raw=raw, validation=report, pre=pre, diagnosis=diag,
        frameworks=frameworks, scoring=scoring, advisory=advisory,
    )
