"""Codex CLI 整合 — 透過 `codex exec --json` 串流回應"""

import asyncio
import json
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path


def is_codex_available() -> bool:
    """檢查 codex CLI 是否已安裝。"""
    return shutil.which("codex") is not None


def get_codex_models() -> list[str]:
    """從 ~/.codex/models_cache.json 讀取可用模型清單。"""
    cache = Path.home() / ".codex" / "models_cache.json"
    if not cache.exists():
        return []
    try:
        data = json.loads(cache.read_text())
        models = []
        for m in data.get("models", []):
            vis = m.get("visibility", "")
            if vis == "list":
                models.append(m["slug"])
        return models
    except (json.JSONDecodeError, KeyError):
        return []


async def codex_login() -> tuple[bool, str]:
    """執行 codex login（開瀏覽器 OAuth）。回傳 (成功, 訊息)。"""
    proc = await asyncio.create_subprocess_exec(
        "codex", "login",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout or b"").decode() + (stderr or b"").decode()
    if proc.returncode == 0:
        return True, output.strip() or "登入成功"
    return False, output.strip() or "登入失敗"


async def codex_login_status() -> str:
    """檢查 codex 登入狀態。"""
    proc = await asyncio.create_subprocess_exec(
        "codex", "login", "status",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    output = (stdout or b"").decode().strip()
    if not output:
        output = (stderr or b"").decode().strip()
    return output or "未知狀態"


async def codex_stream(
    prompt: str,
    system_prompt: str = "",
    model: str = "",
) -> AsyncGenerator[str, None]:
    """透過 codex exec --json 串流回應。

    解析 JSONL 事件，擷取文字輸出。
    支援 codex CLI v0.111+ 的 item.completed 事件格式。
    """
    cmd = ["codex", "exec", "--json"]
    if model:
        cmd.extend(["-m", model])

    # Combine system prompt + user prompt into one input
    # (codex exec treats stdin as the full prompt/instructions)
    full_input = ""
    if system_prompt:
        full_input = f"[System Instructions]\n{system_prompt}\n\n[User Message]\n{prompt}"
    else:
        full_input = prompt

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Send combined prompt via stdin
    if proc.stdin:
        proc.stdin.write(full_input.encode())
        proc.stdin.close()

    if not proc.stdout:
        return

    buffer = b""
    while True:
        chunk = await proc.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode().strip()
            if not line:
                continue

            # Skip non-JSON lines
            if not line.startswith("{"):
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            # v0.111+ format: item.completed with text
            if event_type == "item.completed":
                item = event.get("item", {})
                text = item.get("text", "")
                if text:
                    yield text
            # Streaming delta (older or API format)
            elif event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    yield delta
            # Legacy message format
            elif event_type == "message":
                content = event.get("content", "")
                if content:
                    yield content

    await proc.wait()
