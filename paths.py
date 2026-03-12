"""De-insight 路徑管理。所有資料路徑的唯一來源。"""

import os
import re
from pathlib import Path

__version__ = "1.0.0-pre"

APP_HOME = Path(os.environ.get("DEINSIGHT_HOME", Path.home() / ".deinsight"))
DATA_VERSION = os.environ.get("DEINSIGHT_DATA_VERSION", "v0.7")

DATA_ROOT = APP_HOME / DATA_VERSION
APP_DB = DATA_ROOT / "app.db"
PROJECTS_DIR = DATA_ROOT / "projects"

# 全局文獻庫使用固定 project_id
GLOBAL_PROJECT_ID = "__global__"

def _validate_project_id(project_id: str) -> str:
    """驗證 project_id 不含路徑穿越字元。"""
    if not project_id or ".." in project_id or "/" in project_id or "\\" in project_id:
        if project_id != GLOBAL_PROJECT_ID and not re.match(r'^[a-zA-Z0-9_\-]+$', project_id):
            raise ValueError(f"Invalid project_id: {project_id!r}")
    return project_id


def project_root(project_id: str) -> Path:
    _validate_project_id(project_id)
    return PROJECTS_DIR / project_id


def ensure_project_dirs(project_id: str) -> Path:
    root = project_root(project_id)
    (root / "lancedb").mkdir(parents=True, exist_ok=True)
    (root / "lightrag").mkdir(parents=True, exist_ok=True)
    (root / "documents").mkdir(parents=True, exist_ok=True)
    return root


def app_db_path() -> Path:
    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    return APP_DB
