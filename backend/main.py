"""De-insight 後端入口"""

import glob as _glob
import json
import sys
from pathlib import Path

# Auto-discover venv site-packages so backend works regardless of
# which Python interpreter is used to launch uvicorn.
_venv_dir = Path(__file__).resolve().parent / ".venv"
if _venv_dir.is_dir():
    for _p in _glob.glob(str(_venv_dir / "lib" / "python*" / "site-packages")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

# Ensure project root is importable (for shared modules like config/).
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

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
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers.chat import router as chat_router  # noqa: E402
from routers.images import router as images_router  # noqa: E402
from routers.ingest import router as ingest_router  # noqa: E402

app.include_router(chat_router, prefix="/api")
app.include_router(images_router, prefix="/api")
app.include_router(ingest_router, prefix="/api")

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
    # 安全性：CORS 已限制只接受 localhost 來源
    from config.service import get_config_service
    import services.llm as llm_mod
    from embeddings.service import reset_embedding_service
    from rag.knowledge_graph import reset_rag

    cfg = get_config_service()
    cfg.reload()
    # Export all persisted config keys so downstream libraries (e.g. LiteLLM)
    # that still read process env get a consistent, up-to-date view.
    cfg.export_to_environ()
    llm_mod.DEFAULT_MODEL = cfg.get("LLM_MODEL", "ollama/llama3.2")
    # Also reset long-lived singletons that cache env-sensitive configs.
    reset_embedding_service()
    reset_rag()
    issues = cfg.validate()
    return {
        "status": "ok",
        "model": llm_mod.DEFAULT_MODEL,
        "embed_provider": cfg.get("EMBED_PROVIDER", ""),
        "embed_model": cfg.get("EMBED_MODEL", ""),
        "rag_model": cfg.get("RAG_LLM_MODEL", ""),
        "config_issues": issues,
    }
