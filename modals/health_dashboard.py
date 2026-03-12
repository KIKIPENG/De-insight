"""Health Dashboard Modal — Monitor API/system health."""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

log = logging.getLogger(__name__)


class HealthDashboardModal(ModalScreen):
    """系統健康監測儀板 — 顯示 API 狀態、記憶體、磁碟等資訊。"""

    BINDINGS = [("escape", "close", "關閉")]

    CSS = """
    HealthDashboardModal { align: center middle; }
    #health-box {
        width: 80; height: auto; max-height: 90%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #fafafa;
    }
    #health-scroll {
        height: auto; max-height: 70%;
    }
    .health-section {
        height: auto; margin: 1 0; padding: 0; color: #8b949e;
    }
    .health-title {
        color: #7dd3fc; margin: 0 0 1 0; padding: 0 0 0 0;
    }
    .health-item {
        height: auto; padding: 0 2; color: #8b949e;
    }
    .health-item.-ok {
        color: #4ade80;
    }
    .health-item.-warning {
        color: #facc15;
    }
    .health-item.-error {
        color: #ff6b6b;
    }
    .health-sep {
        height: 1; margin: 1 0; color: #2a2a2a;
    }
    .health-actions { height: auto; margin: 1 0 0 0; }
    .health-btn {
        background: transparent; color: #484f58;
        border: none; height: 1; min-width: 0;
        margin: 0 1 0 0; padding: 0 1;
    }
    .health-btn:hover { color: #fafafa; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._refresh_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        box = Vertical(id="health-box")
        box.border_title = "⚕ 系統健康監測"
        with box:
            with VerticalScroll(id="health-scroll"):
                yield Static("", id="health-content")
            yield Static("[dim #2a2a2a]" + "─" * 76 + "[/]", classes="health-sep")
            with Horizontal(classes="health-actions"):
                yield Button("重新整理", id="health-refresh", classes="health-btn")
            yield Button("← 回到對話", classes="back-btn")

    def on_mount(self) -> None:
        """Refresh health status when modal opens."""
        self._update_health_status()
        self._start_auto_refresh()

    def on_unmount(self) -> None:
        """Stop auto-refresh when modal closes."""
        if self._refresh_task:
            self._refresh_task.cancel()

    @work(exclusive=True, group="health_refresh", thread=False)
    async def _update_health_status(self) -> None:
        """Fetch and display current health status."""
        from settings import load_env
        from utils.health_monitor import get_health_monitor

        try:
            content_parts = []

            # ── LLM Provider & Model ──
            content_parts.append(self._format_section("LLM 服務"))
            env = load_env()
            model = env.get("LLM_MODEL", "未設定")
            content_parts.append(self._format_item(f"模型: {model}", "info"))
            content_parts.append("")

            # ── API Connection Status ──
            content_parts.append(self._format_section("API 連線狀態"))
            api_ok = await self._check_api_connection(env)
            status = "✓ 正常" if api_ok else "✗ 無法連線"
            status_class = "ok" if api_ok else "error"
            content_parts.append(self._format_item(f"後端服務: {status}", status_class))
            content_parts.append("")

            # ── Health Monitor Stats ──
            monitor = get_health_monitor()
            summary = monitor.get_summary()

            content_parts.append(self._format_section("API 呼叫統計"))
            content_parts.append(
                self._format_item(
                    f"本工作階段呼叫: {summary['api_calls_total']} 次",
                    "info",
                )
            )

            # Session duration
            duration = int(summary["session_duration_secs"])
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            secs = duration % 60
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {secs}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {secs}s"
            else:
                duration_str = f"{secs}s"
            content_parts.append(
                self._format_item(f"工作階段時長: {duration_str}", "info")
            )
            content_parts.append("")

            # ── Rate Limiting ──
            content_parts.append(self._format_section("流量限制"))
            rate_limit_count = summary["rate_limits_hit"]
            if rate_limit_count == 0:
                content_parts.append(
                    self._format_item("未觸發流量限制", "ok")
                )
            else:
                content_parts.append(
                    self._format_item(
                        f"觸發流量限制: {rate_limit_count} 次", "warning"
                    )
                )

            if summary["last_rate_limit_at"]:
                when = datetime.fromtimestamp(
                    summary["last_rate_limit_at"]
                ).strftime("%H:%M:%S")
                content_parts.append(
                    self._format_item(f"最後發生: {when}", "warning")
                )
            content_parts.append("")

            # ── Recent Errors ──
            content_parts.append(self._format_section("最近錯誤"))
            errors = summary["api_errors"]
            if not errors:
                content_parts.append(
                    self._format_item("未發生錯誤", "ok")
                )
            else:
                # Show last 5 errors
                for err in errors[-5:]:
                    timestamp = datetime.fromtimestamp(err["timestamp"]).strftime(
                        "%H:%M:%S"
                    )
                    content_parts.append(
                        self._format_item(
                            f"[{timestamp}] {err['type']}: {err['message'][:50]}",
                            "error",
                        )
                    )
            content_parts.append("")

            # ── Database Health ──
            content_parts.append(self._format_section("資料庫連線"))
            db_ok = await self._check_database_health()
            status = "✓ 可存取" if db_ok else "✗ 無法存取"
            status_class = "ok" if db_ok else "error"
            content_parts.append(
                self._format_item(f"記憶資料庫: {status}", status_class)
            )
            content_parts.append("")

            # ── Disk Space ──
            content_parts.append(self._format_section("儲存空間"))
            disk_info = self._check_disk_usage()
            content_parts.append(disk_info)
            content_parts.append("")

            # Update the content widget
            content_text = "\n".join(content_parts)
            self.query_one("#health-content", Static).update(content_text)

        except Exception as e:
            log.exception("Error updating health status")
            error_text = f"[#ff6b6b]無法更新狀態: {str(e)[:100]}[/]"
            try:
                self.query_one("#health-content", Static).update(error_text)
            except Exception:
                pass

    def _format_section(self, title: str) -> str:
        """Format a section title."""
        return f"[#7dd3fc bold]▸ {title}[/]"

    def _format_item(self, text: str, style: str = "info") -> str:
        """Format a status item with color."""
        if style == "ok":
            return f"  [#4ade80]✓ {text}[/]"
        elif style == "warning":
            return f"  [#facc15]⚠ {text}[/]"
        elif style == "error":
            return f"  [#ff6b6b]✗ {text}[/]"
        else:  # info
            return f"  [#8b949e]{text}[/]"

    async def _check_api_connection(self, env: dict) -> bool:
        """Check if backend API is reachable."""
        try:
            import httpx

            api_base = env.get("API_BASE", "http://localhost:8000")
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{api_base}/docs", follow_redirects=False)
                return resp.status_code < 500
        except Exception:
            return False

    async def _check_database_health(self) -> bool:
        """Check if memory database is accessible."""
        try:
            from paths import project_root

            state = getattr(self.app, "state", None)
            if not state or not state.current_project:
                return False

            project_id = state.current_project.get("id")
            if not project_id:
                return False

            root = project_root(project_id)
            db_path = root / "memories.db"
            return db_path.exists() and os.access(db_path, os.R_OK)
        except Exception:
            return False

    def _check_disk_usage(self) -> str:
        """Check disk usage of data directory."""
        try:
            data_dir = Path(__file__).resolve().parent.parent / "data"
            if not data_dir.exists():
                return self._format_item("data 目錄不存在", "warning")

            total_size = 0
            for path in data_dir.rglob("*"):
                if path.is_file():
                    try:
                        total_size += path.stat().st_size
                    except OSError:
                        pass

            # Format size
            if total_size < 1024:
                size_str = f"{total_size} B"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.1f} KB"
            elif total_size < 1024 * 1024 * 1024:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            else:
                size_str = f"{total_size / (1024 * 1024 * 1024):.1f} GB"

            style = "ok" if total_size < 5 * 1024 * 1024 * 1024 else "warning"
            return self._format_item(f"data 目錄: {size_str}", style)
        except Exception as e:
            return self._format_item(f"無法計算: {str(e)[:50]}", "error")

    def _start_auto_refresh(self) -> None:
        """Start auto-refresh timer."""

        @work(exclusive=True, group="health_auto_refresh", thread=False)
        async def _auto_refresh_loop() -> None:
            while True:
                try:
                    await asyncio.sleep(5)
                    await self._update_health_status()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.debug(f"Auto-refresh error: {e}")

        self._refresh_task = asyncio.create_task(_auto_refresh_loop())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "health-refresh":
            asyncio.create_task(self._update_health_status())
        elif event.button.has_class("back-btn"):
            self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
