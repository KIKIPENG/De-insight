"""Chat API 端點：Vercel AI SDK Data Stream Protocol"""

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
                # Vercel AI SDK Data Stream Protocol: text part
                yield f"0:{json.dumps(chunk, ensure_ascii=False)}\n"
            # Finish signal
            yield "d:{}\n"
        except Exception as e:
            error_data = json.dumps(
                {"error": str(e)}, ensure_ascii=False
            )
            yield f"3:{error_data}\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
        },
    )
