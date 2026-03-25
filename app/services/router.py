from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.services import llm_service

IntentType = Literal["chitchat", "rag", "sql"]

PROGRAMS_SCHEMA_HINT = """
Таблица programs содержит информацию об образовательных программах РАНХиГС:
- program       — название программы
- megacluster   — мегакластер (напр. «информационные технологии»)
- institute     — институт (напр. «ИЭМИТ»)
- major         — направление подготовки
- qual          — квалификация (бакалавриат / магистратура)
- edu_form      — форма обучения
- edu_years     — срок обучения (лет)
- pass_2024     — проходной балл 2024
- budget_2025   — бюджетных мест 2025
- contract_2025 — платных мест 2025
- cost          — стоимость обучения (руб/год)
- eges_contract — ЕГЭ для платного поступления
- eges_budget   — ЕГЭ для бюджетного поступления
- tracks        — специализации / треки
- desc          — описание программы
- skills        — компетенции выпускника
"""

SYSTEM_PROMPT = f"""Ты — классификатор намерений для бота поступления в РАНХиГС.
Определи намерение пользователя и верни JSON с полем "intent":

- "sql"      — вопрос требует точных данных из базы программ:
               проходные баллы, количество мест, стоимость, ЕГЭ, сроки, институт, мегакластер.
               {PROGRAMS_SCHEMA_HINT}
- "rag"      — общий вопрос о поступлении, академии, документах, общежитии, FAQ, термины.
- "chitchat" — приветствие, светская беседа, вопрос не по теме поступления.

Отвечай ТОЛЬКО JSON: {{"intent": "sql"}} или {{"intent": "rag"}} или {{"intent": "chitchat"}}
"""


class IntentResult(BaseModel):
    intent: IntentType


def classify(question: str) -> IntentType:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        data = llm_service.chat_json(messages)
        result = IntentResult(**data)
        return result.intent
    except Exception:
        return "rag"
