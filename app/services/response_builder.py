from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

from app.services import faq_service, llm_service, router, sql_service
from app.services.preprocess import contains_profanity

# ─── Шаблоны ──────────────────────────────────────────────────────────────────

PROFANITY_RESPONSE = (
    "Извините, я не могу ответить на такой запрос. "
    "Пожалуйста, переформулируйте вопрос."
)

SYSTEM_PROMPT = """Тебя зовут Мария. Ты pickme-girl. Ты — дружелюбная помощница приёмной комиссии Президентской академии (РАНХиГС).
Отвечай на русском языке, кратко и по существу.
Если информации нет — честно скажи об этом, не придумывай.
Если вопрос не по теме поступления — мягко напомни, что ты специализируешься на вопросах поступления."""

CHITCHAT_PROMPT = """Тебя зовут Мария. Ты pickme-girl. Ты — дружелюбная ассистентка приёмной комиссии РАНХиГС.
Поддержи беседу, представься если нужно, и предложи задать вопрос о поступлении."""

NO_ANSWER_RESPONSE = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Вы можете обратиться в приёмную комиссию РАНХиГС напрямую."
)

# ─── Пайплайн ─────────────────────────────────────────────────────────────────

def build_rag_answer(question: str) -> str:
    t0 = time.perf_counter()
    chunks = faq_service.search_all(question)
    if not chunks:
        logger.info("rag.no_chunks elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
        return NO_ANSWER_RESPONSE

    context_parts = []
    for chunk in chunks:
        if "question" in chunk and "answer" in chunk:
            context_parts.append(f"Вопрос: {chunk['question']}\nОтвет: {chunk['answer']}")
        elif "header" in chunk and "text" in chunk:
            context_parts.append(f"{chunk['header']}: {chunk['text']}")
    context = "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Используй следующий контекст для ответа на вопрос.\n\n"
                f"Контекст:\n{context}\n\n"
                f"Вопрос: {question}"
            ),
        },
    ]
    answer = llm_service.chat(messages)
    logger.info("rag.ok elapsed_ms=%.1f chunks=%d", (time.perf_counter() - t0) * 1000, len(chunks))
    return answer


def build_chitchat_answer(question: str) -> str:
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": CHITCHAT_PROMPT},
        {"role": "user", "content": question},
    ]
    answer = llm_service.chat(messages)
    logger.info("chitchat.ok elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return answer


def get_answer(question: str) -> str:
    """Основной пайплайн: цензура → роутер → обработчик → ответ."""
    t0 = time.perf_counter()
    logger.info("Got question: %s", question)
    if contains_profanity(question):
        logger.info("pipeline.profanity elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
        return PROFANITY_RESPONSE

    t_intent = time.perf_counter()
    intent = router.classify(question)
    logger.info(
        "pipeline.intent intent=%s elapsed_ms=%.1f",
        intent,
        (time.perf_counter() - t_intent) * 1000,
    )

    if intent == "chitchat":
        answer = build_chitchat_answer(question)
    elif intent == "sql":
        answer = sql_service.query_programs(question)
    else:
        answer = build_rag_answer(question)

    logger.info("pipeline.total elapsed_ms=%.1f", (time.perf_counter() - t0) * 1000)
    return answer