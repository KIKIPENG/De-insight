"""De-insight 後端入口"""

import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse as BaseJSONResponse

# 載入專案根目錄的 .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class JSONResponse(BaseJSONResponse):
    """確保中文字符不被 escape。"""

    def render(self, content) -> bytes:
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


app = FastAPI(title="De-insight API", default_response_class=JSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers.chat import router as chat_router  # noqa: E402
from routers.images import router as images_router  # noqa: E402

app.include_router(chat_router, prefix="/api")
app.include_router(images_router, prefix="/api")

# /gallery static — serve frontend/index.html
from fastapi.staticfiles import StaticFiles  # noqa: E402

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.is_dir():
    app.mount("/gallery", StaticFiles(directory=str(_frontend_dir), html=True), name="gallery")


@app.get("/api/health")
async def health():
    from services.llm import get_available_models

    return {"status": "ok", "models": get_available_models()}


@app.post("/api/reload-env")
async def reload_env():
    """重新載入 .env，讓設定變更立即生效。"""
    import os

    import services.llm as llm_mod

    load_dotenv(dotenv_path=env_path, override=True)
    llm_mod.DEFAULT_MODEL = os.getenv("LLM_MODEL", "ollama/llama3.2")
    return {"status": "ok", "model": llm_mod.DEFAULT_MODEL}
