import chromadb
from chromadb.utils import embedding_functions

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SIMILARITY_MAX_DISTANCE,
    TOP_K,
)
from app.models import Source
from app.rag.web_fallback import search_web


def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_relevant_documents(question: str, top_k: int = TOP_K) -> list[Source]:
    """Return document chunks that meet the similarity threshold."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    n_results = min(top_k, collection.count())
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    sources: list[Source] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, distance in zip(documents, metadatas, distances):
        if not doc or not meta:
            continue
        if distance > SIMILARITY_MAX_DISTANCE:
            continue
        sources.append(
            Source(
                filename=meta.get("source", "unknown"),
                page=int(meta.get("page", 0)),
                excerpt=doc[:400] + ("..." if len(doc) > 400 else ""),
                source_type="document",
            )
        )

    return sources


def retrieve_context(
    question: str, force_web: bool = False
) -> tuple[list[Source], bool]:
    """
    Retrieve context for a question.
    Uses uploaded documents first; falls back to web search when none are relevant.
    When force_web is True, web search is used directly, skipping documents.
    Returns (sources, used_web_fallback).
    """
    if not force_web:
        doc_sources = retrieve_relevant_documents(question)
        if doc_sources:
            return doc_sources, False

    web_sources = search_web(question)
    return web_sources, True
