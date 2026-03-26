import os
import logging
from dotenv import load_dotenv

ENV_FILE = os.getenv("ENV_FILE", ".env")
load_dotenv(ENV_FILE)

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO)

from app.api.routes import router as ask_router
from app.db.session import test_connection

app = FastAPI(title="PickMe Ranepa API")

app.include_router(ask_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    return {"db": test_connection()}
