import asyncio
import os

from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()

api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
api_hash = os.getenv("TELEGRAM_API_HASH", "")
bot_username = "@AcademicLLM_bot"

questions = [
    "Как поступить на иканам?",
    "Какие нужны документы?",
    "Есть ли общежитие?",
]
INTERMEDIATE_PHRASES = [
    "обрабатываю ваш запрос",
    "подождите",
    "готовлю ответ",
    "загрузка",
    "thinking",
    "processing",
]

def normalize_text(text: str) -> str:
    return (text or "").strip()

def is_intermediate_message(text: str) -> bool:
    text = normalize_text(text).lower()
    if not text:
        return True
    return any(phrase in text for phrase in INTERMEDIATE_PHRASES)

async def ask_bot(
    client: TelegramClient,
    bot_username: str,
    question: str,
    timeout: int = 180,
    poll_interval: float = 1.0,
    idle_after_answer: float = 3.0,
) -> str:
    entity = await client.get_entity(bot_username)
    bot_id = entity.id

    before_messages = await client.get_messages(bot_username, limit=1)
    before_id = before_messages[0].id if before_messages else 0

    await client.send_message(bot_username, question)

    loop = asyncio.get_running_loop()
    start_time = loop.time()

    collected_parts = []
    seen_message_ids = set()
    last_meaningful_time = None
    last_snapshot = {}

    while loop.time() - start_time < timeout:
        messages = await client.get_messages(bot_username, limit=15)
        messages = sorted(messages, key=lambda m: m.id)

        for msg in messages:
            if msg.id <= before_id:
                continue

            # берем только входящие сообщения от бота
            if msg.out:
                continue

            sender_id = getattr(msg, "sender_id", None)
            if sender_id != bot_id:
                continue

            text = normalize_text(msg.text)
            if not text:
                continue

            prev_text = last_snapshot.get(msg.id)
            last_snapshot[msg.id] = text

            if is_intermediate_message(text):
                if prev_text != text:
                    print(f"[BOT SERVICE] id={msg.id}: {text}")
                continue

            if msg.id not in seen_message_ids:
                seen_message_ids.add(msg.id)
                collected_parts.append(text)
                last_meaningful_time = loop.time()
                print(f"[BOT ANSWER NEW] id={msg.id}: {text}")
                continue

            # если бот отредактировал сообщение
            if prev_text is not None and prev_text != text:
                replaced = False
                for i in range(len(collected_parts) - 1, -1, -1):
                    if collected_parts[i] == prev_text:
                        collected_parts[i] = text
                        replaced = True
                        break

                if not replaced:
                    collected_parts.append(text)

                last_meaningful_time = loop.time()
                print(f"[BOT ANSWER EDIT] id={msg.id}: {text}")

        if collected_parts and last_meaningful_time is not None:
            if loop.time() - last_meaningful_time >= idle_after_answer:
                return "\n\n".join(collected_parts)

        await asyncio.sleep(poll_interval)

    raise TimeoutError("Не дождались финального ответа бота")

async def main():
    client = TelegramClient("session_name", api_id, api_hash)
    await client.start()

    try:
        for q in questions:
            print(f"\n=== ВОПРОС ===\n{q}")
            ans = await ask_bot(client, bot_username, q)
            print(f"\n=== ОТВЕТ ===\n{ans}")
            await asyncio.sleep(2)

    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())