from fastapi import APIRouter
from pydantic import BaseModel

from app.services.response_builder import get_answer

router = APIRouter()


class AskRequest(BaseModel):
    text: str
    user_id: int | None = None


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    answer = get_answer(req.text)
    return AskResponse(answer=answer)
