"""
src/knowledge/ingest.py
=======================
Populate (or refresh) the ChromaDB collection from ``FRAMEWORK_DOCS``.

Idempotent: calling it repeatedly upserts the same IDs. If ChromaDB is not
installed the function returns ``False`` and the retriever uses its keyword
fallback instead.
"""
from __future__ import annotations

from config import CHROMA_COLLECTION, CHROMA_DIR
from src.knowledge.embeddings import get_embedder
from src.knowledge.frameworks import FRAMEWORK_DOCS
from src.utils.logging_config import get_logger

log = get_logger(__name__)


def get_chroma_collection():
    """Return a persistent Chroma collection, or None if Chroma is unavailable."""
    try:
        import chromadb
    except Exception as exc:
        log.info("chromadb not available (%s).", type(exc).__name__)
        return None
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=CHROMA_COLLECTION)


def ingest_frameworks(force: bool = False) -> bool:
    """Embed and upsert the framework corpus into ChromaDB.

    Returns True on success, False if Chroma is unavailable.
    """
    coll = get_chroma_collection()
    if coll is None:
        return False

    if not force and coll.count() >= len(FRAMEWORK_DOCS):
        log.info("Chroma already populated (%d docs).", coll.count())
        return True

    embed = get_embedder()
    ids = [d["id"] for d in FRAMEWORK_DOCS]
    docs = [d["text"] for d in FRAMEWORK_DOCS]
    metas = [{"source": d["source"], "tags": ", ".join(d["tags"])}
             for d in FRAMEWORK_DOCS]
    vectors = embed(docs)

    coll.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
    log.info("Ingested %d framework docs into Chroma.", len(ids))
    return True


if __name__ == "__main__":  # `python -m src.knowledge.ingest`
    ok = ingest_frameworks(force=True)
    print("Ingestion:", "OK (ChromaDB)" if ok else "skipped (keyword fallback)")
