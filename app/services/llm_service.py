from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

import app.config as cfg

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=cfg.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.3,
    response_format: dict | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model or cfg.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format:
        kwargs["response_format"] = response_format

    resp = _get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(messages: list[dict[str, str]], model: str | None = None) -> dict:
    raw = chat(messages, model=model, response_format={"type": "json_object"})
    return json.loads(raw)
