"""
src/knowledge/embeddings.py
===========================
Embedding provider with graceful degradation:

1. Try Ollama (``OLLAMA_EMBED_MODEL``) for real local embeddings.
2. Fall back to a deterministic hashing bag-of-words embedding that needs no
   external service, so the pipeline always runs.

The fallback is *not* semantically strong, but it keeps the system functional
offline and during CI. Install Ollama for production-quality retrieval.
"""
from __future__ import annotations

import hashlib
import re

from config import OLLAMA_EMBED_MODEL, OLLAMA_HOST
from src.utils.logging_config import get_logger

log = get_logger(__name__)

_DIM = 256  # fallback vector dimensionality
_WORD = re.compile(r"[a-zA-Z]+")


class _OllamaEmbedder:
    def __init__(self) -> None:
        import ollama  # imported lazily so absence is non-fatal

        self._client = ollama.Client(host=OLLAMA_HOST)
        # Probe once so we fail fast and fall back if the model is missing.
        self._client.embeddings(model=OLLAMA_EMBED_MODEL, prompt="ping")
        log.info("Using Ollama embeddings: %s", OLLAMA_EMBED_MODEL)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            r = self._client.embeddings(model=OLLAMA_EMBED_MODEL, prompt=t)
            vecs.append(list(r["embedding"]))
        return vecs


class _HashEmbedder:
    """Deterministic, dependency-free fallback embedder (hashed BoW + L2 norm)."""

    def __init__(self) -> None:
        log.warning("Ollama unavailable — using deterministic hash embeddings "
                    "(install Ollama for better retrieval).")

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * _DIM
            for w in _WORD.findall(t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                vec[h % _DIM] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder():
    """Return the best available embedder callable: ``(list[str]) -> list[vec]``."""
    try:
        return _OllamaEmbedder()
    except Exception as exc:  # ollama missing / not running / model absent
        log.info("Ollama embedder not available (%s).", type(exc).__name__)
        return _HashEmbedder()
