from __future__ import annotations

from functools import lru_cache
from typing import Any

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

import app.config as cfg

FAQ_COLLECTION = "faq"
TERMS_COLLECTION = "terms"


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    return SentenceTransformer(cfg.EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_milvus() -> MilvusClient:
    return MilvusClient(uri=cfg.MILVUS_URI)


def _embed(text: str) -> list[float]:
    model = _get_embed_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def _search(
    collection: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    anns_field: str,
) -> list[dict[str, Any]]:
    client = _get_milvus()
    try:
        if not client.has_collection(collection):
            return []
    except Exception:
        return []

    vector = _embed(query)
    try:
        results = client.search(
            collection_name=collection,
            data=[vector],
            anns_field=anns_field,
            limit=top_k,
            output_fields=output_fields,
            search_params={"metric_type": "COSINE", "params": {}},
        )
    except Exception:
        return []

    hits = results[0] if results else []
    return [{"score": h["distance"], **h["entity"]} for h in hits]


def search_faq(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Поиск по FAQ (Database.xlsx): Question + Answer."""
    return _search(
        collection=FAQ_COLLECTION,
        query=query,
        top_k=top_k,
        output_fields=["question", "answer", "question_type"],
        anns_field="question_vector",
    )


def search_terms(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Поиск по терминам (Database-2.xlsx): header + text."""
    return _search(
        collection=TERMS_COLLECTION,
        query=query,
        top_k=top_k,
        output_fields=["header", "text"],
        anns_field="header_vector",
    )


def search_all(query: str, top_k: int = cfg.FAQ_TOP_K) -> list[dict[str, Any]]:
    """Объединённый поиск по обеим коллекциям, сортировка по score."""
    faq = search_faq(query, top_k)
    terms = search_terms(query, top_k)
    combined = faq + terms
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
