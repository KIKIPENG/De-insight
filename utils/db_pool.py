"""aiosqlite 連線池 — 避免頻繁 open/close。

用法：
    async with get_connection(db_path) as db:
        await db.execute(...)
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

log = logging.getLogger("de-insight.db")


class DiskFullError(Exception):
    """Raised when disk is full or I/O error occurs during DB operations."""
    pass

# 每個 db_path 保持一個持久連線
_pool: dict[str, aiosqlite.Connection] = {}
_pool_lock = asyncio.Lock()


async def _get_or_create(db_path: str) -> aiosqlite.Connection:
    """取得或建立一個 aiosqlite 連線。"""
    async with _pool_lock:
        conn = _pool.get(db_path)
        if conn is not None:
            try:
                # 驗證連線是否仍然有效
                await conn.execute("SELECT 1")
                return conn
            except Exception:
                log.warning("DB connection stale, reconnecting: %s", db_path)
                try:
                    await conn.close()
                except Exception:
                    pass
                del _pool[db_path]

        conn = await aiosqlite.connect(db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        _pool[db_path] = conn
        return conn


@asynccontextmanager
async def get_connection(db_path: str | Path) -> AsyncIterator[aiosqlite.Connection]:
    """Context manager: 取得持久連線。不會在退出時關閉。

    Raises DiskFullError if disk is full or I/O errors occur.
    """
    path_str = str(db_path)
    conn = await _get_or_create(path_str)
    try:
        yield conn
    except sqlite3.OperationalError as e:
        err_msg = str(e).lower()
        if any(keyword in err_msg for keyword in ["disk", "full", "i/o"]):
            log.error("Disk full or I/O error on %s: %s", path_str, e)
            raise DiskFullError(f"Disk full or I/O error: {e}") from e
        raise


async def close_all() -> None:
    """關閉所有連線。用於 app shutdown。"""
    async with _pool_lock:
        for path, conn in _pool.items():
            try:
                await conn.close()
                log.info("Closed DB connection: %s", path)
            except Exception as e:
                log.warning("Error closing DB %s: %s", path, e)
        _pool.clear()
