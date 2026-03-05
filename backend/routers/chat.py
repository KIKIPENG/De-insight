"""Chat API 端點：SSE streaming"""

import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from prompts.foucault import get_system_prompt
from services.llm import chat_completion

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    mode: str = "emotional"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("emotional", "rational"):
            raise ValueError("mode 必須是 'emotional' 或 'rational'")
        return v


@router.post("/chat")
async def chat(request: ChatRequest):
    system_prompt = get_system_prompt(request.mode)
    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in request.messages],
    ]

    async def generate():
        try:
            async for chunk in chat_completion(messages):
                data = json.dumps(
                    {"content": chunk, "role": "assistant"},
                    ensure_ascii=False,
                )
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
