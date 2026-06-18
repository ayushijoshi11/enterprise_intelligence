"""
src/agents/framework_retrieval.py
=================================
Agent 2 — Framework Retrieval.

Given the tokens/themes from Agent 1, return the most relevant risk-management
framework chunks.

Backends, auto-selected:
* ChromaBackend  — semantic vector search (Ollama or fallback embeddings).
* KeywordBackend — pure-Python tag/keyword overlap. Always available, so the
  pipeline never hard-fails on a missing dependency.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.knowledge.embeddings import get_embedder
from src.knowledge.frameworks import FRAMEWORK_DOCS
from src.knowledge.ingest import get_chroma_collection, ingest_frameworks
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class Retrieved:
    id: str
    source: str
    text: str
    score: float


class _KeywordBackend:
    """Tag/keyword overlap scoring — no external services required."""

    def __init__(self) -> None:
        self._docs = FRAMEWORK_DOCS
        log.info("Framework retrieval backend: keyword overlap.")

    def query(self, tokens: list[str], k: int) -> list[Retrieved]:
        toks = {t.lower() for t in tokens}
        scored: list[Retrieved] = []
        for d in self._docs:
            tagset = {t.lower() for t in d["tags"]}
            text_low = d["text"].lower()
            tag_overlap = len(toks & tagset)
            text_hits = sum(1 for t in toks if t in text_low)
            score = tag_overlap * 2 + text_hits
            if score > 0:
                scored.append(Retrieved(d["id"], d["source"], d["text"], float(score)))
        scored.sort(key=lambda r: r.score, reverse=True)
        # Always return something useful even on a weak match.
        if not scored:
            scored = [Retrieved(d["id"], d["source"], d["text"], 0.0)
                      for d in self._docs[:k]]
        return scored[:k]


class _ChromaBackend:
    def __init__(self) -> None:
        ingest_frameworks()  # ensure populated
        self._coll = get_chroma_collection()
        if self._coll is None:
            raise RuntimeError("Chroma collection unavailable")
        self._embed = get_embedder()
        log.info("Framework retrieval backend: ChromaDB.")

    def query(self, tokens: list[str], k: int) -> list[Retrieved]:
        query_text = " ".join(tokens) or "risk management logistics"
        qvec = self._embed([query_text])[0]
        res = self._coll.query(query_embeddings=[qvec], n_results=k)
        out: list[Retrieved] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[None] * len(ids)])[0]
        for i, _id in enumerate(ids):
            dist = dists[i] if dists and dists[i] is not None else 1.0
            out.append(Retrieved(
                id=_id,
                source=metas[i].get("source", _id),
                text=docs[i],
                score=round(1.0 / (1.0 + dist), 4),  # similarity-ish
            ))
        return out


class FrameworkRetriever:
    """Public facade. Picks the best backend at construction time."""

    def __init__(self, prefer_chroma: bool = True) -> None:
        self.backend = None
        if prefer_chroma:
            try:
                self.backend = _ChromaBackend()
            except Exception as exc:
                log.info("Chroma backend unavailable (%s); falling back.",
                         type(exc).__name__)
        if self.backend is None:
            self.backend = _KeywordBackend()

    def retrieve(self, tokens: list[str], k: int = 4) -> list[Retrieved]:
        if not tokens:
            tokens = ["risk", "logistics", "general"]
        results = self.backend.query(tokens, k)
        log.info("Retrieved %d frameworks for %d tokens.", len(results), len(tokens))
        return results
