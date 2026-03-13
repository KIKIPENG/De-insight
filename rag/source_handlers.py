"""Source handlers — 包裝既有 insert_* 函式，加上錯誤分類。"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    title: str = ""
    warning: str = ""
    file_size: int = 0
    page_count: int = 0
    source_type: str = ""
    extra: dict = field(default_factory=dict)


class SourceHandler(ABC):
    @abstractmethod
    async def ingest(self, job: dict) -> IngestResult:
        """Execute ingestion for a single job. Raises on failure."""
        ...


class PdfIngestHandler(SourceHandler):
    async def ingest(self, job: dict) -> IngestResult:
        from rag.knowledge_graph import insert_pdf

        meta = await insert_pdf(
            job["source"],
            project_id=job["project_id"],
            title=job.get("title", ""),
            job_id=job.get("id"),
        )
        return IngestResult(
            title=meta["title"],
            warning=meta.get("warning", ""),
            file_size=meta.get("file_size", 0),
            page_count=meta.get("page_count", 0),
            source_type="pdf",
        )


class UrlIngestHandler(SourceHandler):
    async def ingest(self, job: dict) -> IngestResult:
        from rag.knowledge_graph import insert_url

        meta = await insert_url(
            job["source"],
            project_id=job["project_id"],
            title=job.get("title", ""),
            job_id=job.get("id"),
        )
        return IngestResult(
            title=meta["title"],
            warning=meta.get("warning", ""),
            file_size=meta.get("file_size", 0),
            page_count=meta.get("page_count", 0),
            source_type="url",
        )


class TextIngestHandler(SourceHandler):
    async def ingest(self, job: dict) -> IngestResult:
        import json

        from rag.knowledge_graph import insert_text

        payload = json.loads(job.get("payload_json", "{}"))
        content = payload.get("content", "")
        title = job.get("title", "") or "手動貼上文獻"
        warning = await insert_text(content, source=title, project_id=job["project_id"], job_id=job.get("id"))
        return IngestResult(
            title=title,
            warning=warning or "",
            file_size=len(content.encode("utf-8")),
            page_count=0,
            source_type="text",
        )


class DoiIngestHandler(SourceHandler):
    async def ingest(self, job: dict) -> IngestResult:
        from rag.knowledge_graph import insert_doi

        meta = await insert_doi(
            job["source"],
            project_id=job["project_id"],
            title=job.get("title", ""),
            job_id=job.get("id"),
        )
        return IngestResult(
            title=meta["title"],
            warning=meta.get("warning", ""),
            file_size=meta.get("file_size", 0),
            page_count=meta.get("page_count", 0),
            source_type="doi",
        )


class TxtFileIngestHandler(SourceHandler):
    """從磁碟讀取 .txt 檔案，呼叫 insert_text() 匯入知識庫。"""

    async def ingest(self, job: dict) -> IngestResult:
        import os

        from rag.knowledge_graph import insert_text

        path = job["source"]
        title = job.get("title", "") or os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"檔案內容為空: {path}")
        warning = await insert_text(content, source=title, project_id=job["project_id"], job_id=job.get("id"))
        return IngestResult(
            title=title,
            warning=warning or "",
            file_size=len(content.encode("utf-8")),
            page_count=0,
            source_type="txt",
        )


class MdFileIngestHandler(SourceHandler):
    """從磁碟讀取 .md 檔案，呼叫 insert_text() 匯入知識庫。"""

    async def ingest(self, job: dict) -> IngestResult:
        import os

        from rag.knowledge_graph import insert_text

        path = job["source"]
        title = job.get("title", "") or os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"檔案內容為空: {path}")
        warning = await insert_text(content, source=title, project_id=job["project_id"], job_id=job.get("id"))
        return IngestResult(
            title=title,
            warning=warning or "",
            file_size=len(content.encode("utf-8")),
            page_count=0,
            source_type="md",
        )


class MovementJsonIngestHandler(SourceHandler):
    """匯入 movement JSON：知識性內容進 LightRAG，persona 存全局。"""

    async def ingest(self, job: dict) -> IngestResult:
        import json as _json
        import os

        from rag.knowledge_graph import insert_text
        from persona.store import (
            _is_valid_movement,
            extract_persona_from_movement,
            extract_knowledge_text,
            save_persona,
        )

        path = job["source"]
        title = job.get("title", "") or os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            movement = _json.load(f)

        # Fix #5: 驗證 JSON 是否為有效的 movement 格式
        if not _is_valid_movement(movement):
            raise ValueError(
                f"JSON 檔案不是有效的 movement 格式（缺少 movement_id 或 judge_persona_seed）: {path}"
            )

        mid = movement.get("movement_id", os.path.splitext(os.path.basename(path))[0])
        name_zh = movement.get("name", {}).get("zh", mid)

        # 1. 抽取 persona → 存全局
        persona_data = extract_persona_from_movement(movement)
        save_persona(mid, persona_data)

        # 2. 抽取知識文本 → LightRAG
        knowledge_md = extract_knowledge_text(movement)
        if not knowledge_md.strip():
            raise ValueError(f"Movement JSON 無知識性內容: {path}")

        warning = await insert_text(
            knowledge_md,
            source=f"movement:{name_zh}",
            project_id=job["project_id"],
            job_id=job.get("id"),
        )
        return IngestResult(
            title=f"{name_zh}（流派知識 + 批評視角）",
            warning=warning or "",
            file_size=len(knowledge_md.encode("utf-8")),
            page_count=0,
            source_type="movement_json",
            extra={"persona_id": mid, "persona_name": name_zh},
        )


HANDLER_MAP: dict[str, type[SourceHandler]] = {
    "pdf": PdfIngestHandler,
    "url": UrlIngestHandler,
    "text": TextIngestHandler,
    "txt": TxtFileIngestHandler,
    "md": MdFileIngestHandler,
    "doi": DoiIngestHandler,
    "movement_json": MovementJsonIngestHandler,
}


def get_handler(source_type: str) -> SourceHandler:
    cls = HANDLER_MAP.get(source_type)
    if cls is None:
        raise ValueError(f"Unknown source_type: {source_type}")
    return cls()
