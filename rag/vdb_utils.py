"""Helpers for locating LightRAG vector DB metadata files."""

from __future__ import annotations

from pathlib import Path

import paths

VDB_CHUNKS_FILE_NAMES = ("vdb_chunks.json", "vdb_chunks_vdb.json")


def get_lightrag_dir(project_id: str = "default") -> Path:
    if project_id == "default":
        return paths.DATA_ROOT / "projects" / "default" / "lightrag"
    return paths.project_root(project_id) / "lightrag"


def list_vdb_chunk_files(working_dir: Path) -> list[Path]:
    return [working_dir / name for name in VDB_CHUNKS_FILE_NAMES]


def find_vdb_chunks_file(project_id: str = "default") -> Path | None:
    working_dir = get_lightrag_dir(project_id)
    for path in list_vdb_chunk_files(working_dir):
        if path.exists():
            return path
    return None
