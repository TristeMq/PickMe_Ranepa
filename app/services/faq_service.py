from __future__ import annotations

from functools import lru_cache
import logging
import time
from typing import Any

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

import app.config as cfg

logger = logging.getLogger(__name__)
FAQ_COLLECTION = "faq"
TERMS_COLLECTION = "terms"


@lru_cache(maxsize=1)
def _get_embed_model() -> SentenceTransformer:
    t0 = time.perf_counter()
    logger.info("embed.model_load_start name=%s", cfg.EMBED_MODEL)
    model = SentenceTransformer(cfg.EMBED_MODEL)
    logger.info("embed.model_load_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return model


@lru_cache(maxsize=1)
def _get_milvus() -> MilvusClient:
    t0 = time.perf_counter()
    logger.info("milvus.client_init_start uri=%s", cfg.MILVUS_URI)
    client = MilvusClient(uri=cfg.MILVUS_URI)
    logger.info("milvus.client_init_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return client


def _embed(text: str) -> list[float]:
    t0 = time.perf_counter()
    model = _get_embed_model()
    vec = model.encode(text, normalize_embeddings=True).tolist()
    logger.info("embed.encode_ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return vec


def _search(
    collection: str,
    query: str,
    top_k: int,
    output_fields: list[str],
    anns_field: str,
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    logger.info("milvus.search_start name=%s top_k=%d", collection, top_k)
    client = _get_milvus()
    try:
        if not client.has_collection(collection):
            logger.info("milvus.no_collection name=%s elapsed_ms=%.1f", collection, (time.perf_counter() - t0) * 1000)
            return []
    except Exception:
        logger.exception("milvus.has_collection_failed name=%s", collection)
        return []

    logger.info("embed.start")
    vector = _embed(query)
    logger.info("embed.done")
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
        logger.exception("milvus.search_failed name=%s", collection)
        return []

    hits = results[0] if results else []
    out = [{"score": h["distance"], **h["entity"]} for h in hits]
    logger.info(
        "milvus.search_ok name=%s hits=%d elapsed_ms=%.1f",
        collection,
        len(out),
        (time.perf_counter() - t0) * 1000,
    )
    return out


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
