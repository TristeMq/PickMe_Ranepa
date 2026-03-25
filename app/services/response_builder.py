from __future__ import annotations

from app.services import faq_service, llm_service, router, sql_service
from app.services.preprocess import contains_profanity

# ─── Шаблоны ──────────────────────────────────────────────────────────────────

PROFANITY_RESPONSE = (
    "Извините, я не могу ответить на такой запрос. "
    "Пожалуйста, переформулируйте вопрос."
)

SYSTEM_PROMPT = """Ты — дружелюбный помощник приёмной комиссии Президентской академии (РАНХиГС).
Отвечай на русском языке, кратко и по существу.
Если информации нет — честно скажи об этом, не придумывай.
Если вопрос не по теме поступления — мягко напомни, что ты специализируешься на вопросах поступления."""

CHITCHAT_PROMPT = """Ты — дружелюбный ассистент приёмной комиссии РАНХиГС.
Поддержи беседу, представься если нужно, и предложи задать вопрос о поступлении."""

NO_ANSWER_RESPONSE = (
    "К сожалению, у меня нет точной информации по этому вопросу. "
    "Вы можете обратиться в приёмную комиссию РАНХиГС напрямую."
)

# ─── Пайплайн ─────────────────────────────────────────────────────────────────

def build_rag_answer(question: str) -> str:
    chunks = faq_service.search_all(question)
    if not chunks:
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
    return llm_service.chat(messages)


def build_chitchat_answer(question: str) -> str:
    messages = [
        {"role": "system", "content": CHITCHAT_PROMPT},
        {"role": "user", "content": question},
    ]
    return llm_service.chat(messages)


def get_answer(question: str) -> str:
    """Основной пайплайн: цензура → роутер → обработчик → ответ."""
    if contains_profanity(question):
        return PROFANITY_RESPONSE

    intent = router.classify(question)

    if intent == "chitchat":
        return build_chitchat_answer(question)
    elif intent == "sql":
        return sql_service.query_programs(question)
    else:
        return build_rag_answer(question)
