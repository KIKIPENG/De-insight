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
        warning = await insert_text(content, source=title, project_id=job["project_id"])
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
        )
        return IngestResult(
            title=meta["title"],
            warning=meta.get("warning", ""),
            file_size=meta.get("file_size", 0),
            page_count=meta.get("page_count", 0),
            source_type="doi",
        )


HANDLER_MAP: dict[str, type[SourceHandler]] = {
    "pdf": PdfIngestHandler,
    "url": UrlIngestHandler,
    "text": TextIngestHandler,
    "doi": DoiIngestHandler,
}


def get_handler(source_type: str) -> SourceHandler:
    cls = HANDLER_MAP.get(source_type)
    if cls is None:
        raise ValueError(f"Unknown source_type: {source_type}")
    return cls()
