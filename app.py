"""De-insight TUI — App 主體"""

import asyncio
import sys
from pathlib import Path

# Allow importing from backend/
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Auto-discover backend venv site-packages so TUI works regardless of
# which Python interpreter is used to launch it.
_venv_dir = Path(__file__).parent / "backend" / ".venv"
if _venv_dir.is_dir():
    import glob as _glob
    _sp = _glob.glob(str(_venv_dir / "lib" / "python*" / "site-packages"))
    for _p in _sp:
        if _p not in sys.path:
            sys.path.insert(1, _p)

from widgets import (
    AppState, ChatInput, MenuBar, WelcomeBlock,
)
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import OptionList

from panels import MemoryPanel, ResearchPanel
from conversation.store import ConversationStore
from projects.manager import ProjectManager

from mixins.chat import ChatMixin
from mixins.memory import MemoryMixin
from mixins.rag import RAGMixin
from mixins.project import ProjectMixin
from mixins.ui import UIMixin


class DeInsightApp(ChatMixin, MemoryMixin, RAGMixin, ProjectMixin, UIMixin, App):
    TITLE = "De-insight"
    CSS = """
    Screen {
        background: #0a0a0a;
        color: #fafafa;
    }

    /* ── menu bar ── */
    MenuBar {
        dock: top;
        height: auto;
        padding: 0 1;
        background: #0d0d0d;
        color: #6e7681;
        border-bottom: solid #1e1e1e;
    }

    MenuBar:hover {
        color: #fafafa;
    }

    /* ── chatbox actions ── */
    .chatbox-actions {
        height: 1;
        margin: 0;
        padding: 0;
    }

    ActionLink {
        width: auto;
        height: 1;
        padding: 0;
        margin: 0 1 0 0;
        color: #2a2a2a;
    }

    ActionLink:hover {
        color: #d4a27a;
    }

    /* ── chat scroll area ── */
    #chat-scroll {
        background: #0a0a0a;
        scrollbar-size: 1 1;
        scrollbar-color: #2a2a2a;
        scrollbar-color-hover: #484f58;
        scrollbar-color-active: #6e7681;
    }

    #messages {
        padding: 1 2;
        height: auto;
    }

    /* ── welcome ── */
    WelcomeBlock {
        padding: 1 2;
        margin: 0 1;
        height: auto;
        border: round #1e1e1e;
        border-title-color: #d4a27a;
    }

    /* ── chatbox ── */
    Chatbox {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #1e1e1e;
        border-title-color: #484f58;
    }

    .chatbox-user {
        border: round #2a2a2a;
        border-title-color: #6e7681;
    }

    .chatbox-assistant {
        border: round #2a2a2a;
        border-title-color: #d4a27a;
    }

    .chatbox-assistant.responding {
        background: #d4a27a 4%;
        border-title-color: #d4a27a;
    }

    .chatbox-body {
        margin: 0;
        padding: 0;
        height: auto;
        color: #fafafa;
    }

    .chatbox-user .chatbox-body {
        color: #c9d1d9;
    }

    .stream-body {
        margin: 0;
        padding: 0;
        height: auto;
        color: #fafafa;
    }

    /* ── markdown overrides ── */
    Markdown {
        margin: 0;
        padding: 0 1;
        background: transparent;
    }

    Markdown MarkdownH1 {
        color: #f0f6fc;
        text-style: bold;
        border-bottom: solid #30363d;
        margin-bottom: 1;
        padding: 0;
        background: transparent;
    }

    Markdown MarkdownH2 {
        color: #c9d1d9;
        text-style: bold;
        margin-top: 1;
        padding: 0;
        background: transparent;
    }

    Markdown MarkdownH3 {
        color: #8b949e;
        text-style: bold italic;
        padding: 0;
        background: transparent;
    }

    MarkdownFence {
        margin: 1 0;
        padding: 0 1;
        background: #161b22;
        border-left: thick #30363d;
        color: #e6edf3;
    }

    MarkdownBlockQuote {
        margin: 0;
        padding: 0 1;
        border-left: thick #30363d;
        background: transparent;
        color: #8b949e;
    }

    MarkdownBulletList, MarkdownOrderedList {
        margin: 0;
        padding: 0 0 0 2;
    }

    Markdown .inline-code {
        color: #7dd3fc;
        background: #1e2d3d;
    }

    Markdown MarkdownHorizontalRule {
        color: #30363d;
    }

    /* ── knowledge concept link ── */
    ActionLink.source-link {
        color: #484f58;
    }
    ActionLink.source-link:hover {
        color: #d4a27a;
    }

    /* ── thinking indicator ── */
    ThinkingIndicator {
        height: 1;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #d4a27a 50%;
        border-title-color: #d4a27a;
    }

    /* ── input area ── */
    #input-box {
        dock: bottom;
        height: auto;
        max-height: 16;
        padding: 0 1;
        margin: 0;
        background: #0a0a0a;
    }

    #input-frame {
        height: auto;
        margin: 0 1;
        padding: 0 1;
        border: round #2a2a2a;
        border-title-color: #484f58;
        background: #0d0d0d;
    }

    #input-frame:focus-within {
        border: round #d4a27a 70%;
        border-title-color: #d4a27a;
    }

    #chat-input {
        background: transparent;
        color: #fafafa;
        border: none;
        padding: 0;
        margin: 0;
        height: auto;
        min-height: 3;
        max-height: 14;
    }

    /* ── layout ── */
    #main-horizontal {
        height: 1fr;
    }

    #chat-column {
        width: 1fr;
    }

    #right-panel {
        width: 35%;
        border-left: solid #2a2a2a;
    }

    /* ── research panel ── */
    #research-panel {
        height: 80%;
        border-bottom: solid #2a2a2a;
        border-top: solid #2a2a2a;
        border-title-color: #6e7681;
        padding: 0 1;
    }

    .kb-actions-row {
        height: auto;
        margin: 1 0 1 0;
    }
    .kb-action {
        width: auto;
        min-width: 0;
        padding: 0 1;
        background: transparent;
        color: #6e7681;
        border: none;
    }
    .kb-action:hover {
        color: #fafafa;
    }
    .kb-action.-active {
        color: #fafafa;
        background: #1a1a1a;
    }
    .kb-action:focus {
        color: #fafafa;
        background: #1a1a1a;
    }
    .kb-divider {
        height: 1;
        color: #2a2a2a;
        margin: 0 0 1 0;
    }
    .kb-section-label {
        margin: 0 0 1 0;
    }
    #kb-doc-scroll {
        height: 25%;
        border: round #2a2a2a;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    .doc-item {
        height: auto;
        padding: 0 1;
        color: #8b949e;
    }
    .doc-item:hover {
        color: #fafafa;
        background: #111111;
    }
    #research-result-scroll {
        height: 75%;
        border: round #2a2a2a;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    #research-content {
        height: auto;
        color: #c9d1d9;
    }
    #research-cite {
        width: auto;
        min-width: 0;
        background: transparent;
        border: none;
        color: #6e7681;
        padding: 0;
    }
    #research-cite:hover {
        color: #fafafa;
    }
    #research-cite:disabled {
        color: #484f58;
    }

    /* ── focus panel ── */
    #research-view {
        height: 100%;
    }
    #focus-view {
        display: none;
        height: 100%;
        overflow-y: auto;
        padding: 1 1;
    }
    .focus-label {
        color: #6e7681;
        margin: 1 0 0 0;
        height: 1;
    }
    .focus-editor {
        height: 1fr;
        min-height: 12;
        margin: 0 0 1 0;
        border: solid #2a2a2a;
        background: #0d0d0d;
    }
    #focus-buttons {
        height: auto;
        margin-top: 1;
    }
    #focus-buttons Button {
        height: 1;
        min-width: 10;
        margin-right: 1;
        background: #1a1a1a;
        color: #8b949e;
        border: solid #2a2a2a;
    }
    #focus-buttons Button:hover {
        background: #2a2a2a;
        color: #fafafa;
    }
    #focus-buttons #btn-focus-evaluate {
        border: solid #d4a27a 50%;
        color: #d4a27a;
    }
    #focus-buttons #btn-focus-evaluate:hover {
        background: #3d2a1a;
    }
    #focus-import-status {
        margin-top: 1;
        color: #6e7681;
        height: auto;
    }

    /* ── memory panel ── */
    #memory-panel {
        height: 20%;
        border-top: solid #2a2a2a;
        border-title-color: #6e7681;
        padding: 0 1;
    }

    .memory-item {
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        color: #6e7681;
    }

    .memory-item.-new {
        color: #d4a27a;
        background: #3d2a1a;
    }

    .memory-item:hover {
        color: #fafafa;
        background: #1a1a1a;
    }

    .memory-item.-evolution {
        border-left: solid #d4a27a;
        padding-left: 1;
        color: #d4a27a;
    }

    .memory-item.-contradiction {
        border-left: solid #ff6b6b;
        padding-left: 1;
        color: #ff6b6b;
    }
    .memory-item.-focus-tagged {
        color: #d4a27a;
    }

    /* ── history entries (WelcomeBlock) ── */
    .history-entry {
        height: 1;
        padding: 0 1;
        color: #8b949e;
    }
    .history-entry:hover {
        color: #fafafa;
        background: #111111;
    }

    /* ── slash hint popup ── */
    #slash-hints {
        display: none;
        dock: bottom;
        layer: overlay;
        height: auto;
        max-height: 12;
        width: 40;
        margin: 0 0 0 3;
        background: #1a1a1a;
        border: round #3a3a3a;
        color: #c9d1d9;
        scrollbar-size: 1 1;
        scrollbar-color: #2a2a2a;
    }
    #slash-hints > .option-list--option-highlighted {
        background: #2a2a2a;
        color: #fafafa;
    }
    #slash-hints.-visible {
        display: block;
    }

    /* ── back button (shared across modals) ── */
    .back-btn {
        background: transparent;
        color: #484f58;
        border: none;
        height: 1;
        min-width: 0;
        margin: 1 0 0 0;
        padding: 0 1;
    }
    .back-btn:hover {
        color: #fafafa;
    }

    """

    BINDINGS = [
        Binding("ctrl+s", "open_settings", "設定", show=False, priority=True),
        Binding("ctrl+e", "toggle_mode", "感性/理性", show=False, priority=True),
        Binding("ctrl+n", "new_chat", "新對話", show=False, priority=True),
        Binding("ctrl+k", "search_knowledge", "搜尋知識庫", show=False, priority=True),
        Binding("ctrl+f", "import_document", "匯入文件", show=False, priority=True),
        Binding("ctrl+m", "manage_memories", "記憶管理", show=False, priority=True),
        Binding("ctrl+p", "open_project_modal", "專案管理", show=False, priority=True),
        Binding("ctrl+d", "manage_documents", "文獻管理", show=False, priority=True),
        Binding("ctrl+g", "view_relations", "記憶關聯", show=False, priority=True),
        Binding("ctrl+l", "open_gallery", "圖片庫", show=False, priority=True),
        Binding("ctrl+u", "update_document", "更新文件", show=False, priority=True),
        Binding("ctrl+c", "quit", "退出", show=False),
    ]

    mode: reactive[str] = reactive("emotional")
    is_loading: reactive[bool] = reactive(False)
    rag_mode: str = "fast"

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[dict] = []
        self.api_base = "http://localhost:8000"
        self.state = AppState()
        self._unrelated_insight_count: int = 0
        self._pending_focus_nudge: bool = False
        self._focus_tagged_insights: set[str] = set()
        self._project_manager = ProjectManager()
        self._conv_store = ConversationStore()

    def notify(
        self,
        message,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float = 5,
        **kwargs,
    ) -> None:
        """Route all notifications to the MenuBar right side instead of toast popups."""
        sev_map = {"information": "info", "warning": "warning", "error": "error"}
        sev = sev_map.get(severity, "info")
        try:
            menu = self.query_one("#menu-bar", MenuBar)
            menu.show_message(str(message), severity=sev, timeout=timeout)
        except Exception:
            super().notify(message, title=title, severity=severity, timeout=timeout, **kwargs)

    def compose(self) -> ComposeResult:
        yield MenuBar(id="menu-bar")
        with Horizontal(id="main-horizontal"):
            with Vertical(id="chat-column"):
                yield VerticalScroll(
                    Vertical(id="messages"),
                    id="chat-scroll",
                )
                yield OptionList(id="slash-hints")
                ta = ChatInput(id="chat-input")
                ta.show_line_numbers = False
                input_frame = Vertical(ta, id="input-frame")
                input_frame.border_title = "⌨ Message"
                yield Vertical(input_frame, id="input-box")
            with Vertical(id="right-panel"):
                rp = ResearchPanel(id="research-panel")
                rp.border_title = "◇ Research 研究面板"
                yield rp
                mp = MemoryPanel(id="memory-panel")
                mp.border_title = "◇ Memory 記憶面板"
                yield mp

    async def on_mount(self) -> None:
        from settings import load_env
        projects = []
        try:
            projects = await self._project_manager.list_projects()
        except Exception:
            pass

        env = load_env()
        llm_model = env.get("LLM_MODEL", "").strip()
        needs_onboarding = (not projects) or (not llm_model)
        if needs_onboarding:
            from modals import OnboardingScreen
            await self.push_screen(OnboardingScreen(), callback=self._on_onboarding_done)
            return

        await self._init_app(projects)

    async def _on_onboarding_done(self, result: str | None) -> None:
        """Onboarding 完成後初始化。"""
        projects = await self._project_manager.list_projects()
        if not projects:
            proj = await self._project_manager.create_project("My Project")
            projects = [proj]
        await self._init_app(projects)

    async def _init_app(self, projects: list[dict]) -> None:
        """正常初始化流程（onboarding 後或直接啟動）。"""
        # 開啟軟體時一律先用快速檢索模式。
        self.rag_mode = "fast"
        if projects:
            self.state.current_project = projects[0]
            from paths import project_root, ensure_project_dirs
            from conversation.store import ConversationStore
            pid = projects[0]["id"]
            ensure_project_dirs(pid)
            self._conv_store = ConversationStore(
                db_path=project_root(pid) / "conversations.db"
            )
        # A3: Startup health check — validate RAG config + probe endpoint
        try:
            from rag.pipeline import startup_health_check, is_degraded, get_degraded_reason
            hc = await startup_health_check(probe_llm=True)
            if not hc["healthy"]:
                self.notify(
                    f"知識庫降級模式：{get_degraded_reason()[:80]}",
                    severity="warning",
                    timeout=8,
                )
        except Exception as e:
            self.log.warning(f"Startup health check failed: {e}")

        self._update_menu_bar()
        self._update_status()
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "[#d4a27a]◈[/] De-insight v0.8"
        await container.mount(welcome)
        self.query_one("#chat-input", ChatInput).focus()
        self._refresh_memory_panel()
        self._refresh_knowledge_panel()
        self._reindex_pending_memories()

        # Start ingestion worker + polling timer
        try:
            from rag.ingestion_service import get_ingestion_service
            svc = get_ingestion_service()
            await svc.ensure_table()
            svc.ensure_worker_running()
            self._ingestion_poll_timer = self.set_interval(3.0, self._poll_ingestion_jobs)
        except Exception as e:
            self.log.warning(f"Failed to start ingestion worker: {e}")

        # 定期健康檢查（30 秒）
        self.set_interval(30, self._periodic_health_check)

    async def _periodic_health_check(self) -> None:
        """背景：定期更新服務狀態。"""
        try:
            self._update_status()
        except Exception:
            pass

    async def _poll_ingestion_jobs(self) -> None:
        """Poll for completed/failed/active ingestion jobs and update MenuBar."""
        try:
            from rag.ingestion_service import get_ingestion_service
            svc = get_ingestion_service()

            # Check completed jobs
            completed = await svc.poll_completed()
            for job in completed:
                title = job.get("title") or job.get("source", "")[:40]
                status = job.get("status", "done")
                if status == "done_with_warning":
                    warning = job.get("last_error", "")[:60]
                    self.notify(f"匯入完成（部分警告: {warning}）：{title}")
                else:
                    self.notify(f"匯入完成：{title}")
                self._refresh_knowledge_panel()

            # Check terminal failures (only report each once)
            if not hasattr(self, "_reported_failed_jobs"):
                self._reported_failed_jobs: set[str] = set()
            failed = await svc.poll_failed_terminal()
            for job in failed:
                jid = job["id"]
                if jid in self._reported_failed_jobs:
                    continue
                self._reported_failed_jobs.add(jid)
                title = job.get("title") or job.get("source", "")[:40]
                error = job.get("last_error", "")[:80]
                self.notify(
                    f"匯入失敗：{title}（{error}）",
                    severity="error",
                    timeout=15,
                )

            # Show real progress for active jobs
            active_jobs = await svc.get_active_progress()
            running_jobs = [j for j in active_jobs if str(j.get("status", "")).startswith("running")]
            queued_jobs = [j for j in active_jobs if j.get("status") == "queued"]
            if running_jobs:
                # Prefer the most recently updated running job to avoid stale display.
                running_jobs.sort(key=lambda j: (j.get("updated_at") or "", j.get("created_at") or ""), reverse=True)
                job = running_jobs[0]
                pct = job.get("progress_pct", 0) or 0
                chunks_done = job.get("chunks_done", 0) or 0
                chunks_total = job.get("chunks_total", 0) or 0
                stage = (job.get("progress_stage", "") or "").strip() or "處理中"
                status = str(job.get("status", "") or "")
                phase = (job.get("phase", "") or "").strip()
                import os.path as _osp
                _raw_title = job.get("title") or _osp.basename(job.get("source", ""))
                title = _raw_title[:15] + ("…" if len(_raw_title) > 15 else "")
                detail = None
                try:
                    detail = await svc.get_job_detail(str(job.get("id", "")))
                except Exception:
                    detail = None

                # ETA: use backend estimator only (prevents client-side over-estimation drift).
                eta_str = ""
                eta_seconds = None
                if detail and isinstance(detail.get("eta_seconds"), (int, float)):
                    eta_seconds = detail.get("eta_seconds")
                elif isinstance(job.get("eta_seconds"), (int, float)):
                    eta_seconds = job.get("eta_seconds")
                if isinstance(eta_seconds, (int, float)) and eta_seconds >= 0:
                    remaining = int(eta_seconds)
                    if remaining > 60:
                        eta_str = f" ~{remaining // 60}m{remaining % 60:02d}s"
                    else:
                        eta_str = f" ~{remaining}s"

                if "waiting_backoff" in status:
                    msg = f"速率限制重試 {title}{eta_str}"
                elif detail and isinstance(detail.get("phase_detail"), dict):
                    pd = detail.get("phase_detail") or {}
                    bc = pd.get("batch_current")
                    bt = pd.get("batch_total")
                    pr = pd.get("page_range")
                    if phase == "chunking":
                        msg = f"分塊：頁 {chunks_done}/{chunks_total}{eta_str}"
                    elif phase == "extracting":
                        if bc and bt:
                            if pr:
                                msg = f"抽取：批 {bc}/{bt}（頁 {pr}）{eta_str}"
                            else:
                                msg = f"抽取：批 {bc}/{bt}{eta_str}"
                        else:
                            msg = f"抽取：{title}{eta_str}"
                    elif phase == "merging":
                        if bc and bt:
                            msg = f"合併：批 {bc}/{bt}{eta_str}"
                        else:
                            msg = f"合併：{title}{eta_str}"
                    elif phase == "flushing":
                        if bc and bt:
                            msg = f"寫入：批 {bc}/{bt}{eta_str}"
                        else:
                            msg = f"寫入中：{title}{eta_str}"
                    else:
                        msg = f"{stage} {title}{eta_str}"
                elif chunks_total > 0:
                    msg = f"{stage} {title} {chunks_done}/{chunks_total}{eta_str}"
                else:
                    msg = f"{stage} {title}{eta_str}"

                progress_float = min(max(pct / 100.0, 0.0), 1.0)
                if len(queued_jobs) > 0:
                    msg += f"  +{len(queued_jobs)}等候"

                try:
                    menu = self.query_one("#menu-bar", MenuBar)
                    menu.show_progress(msg, progress_float)
                except Exception:
                    pass
            elif queued_jobs:
                try:
                    menu = self.query_one("#menu-bar", MenuBar)
                    if not svc.worker_alive:
                        menu.show_progress("worker recovering（< 20s）")
                    else:
                        menu.show_progress(f"等候建圖 {len(queued_jobs)} 件")
                except Exception:
                    pass
            else:
                # Show deferred retry countdown (e.g. rate-limit backoff) when no active job.
                deferred = await svc.get_deferred_retries()
                if deferred:
                    d0 = deferred[0]
                    import os.path as _osp
                    _raw_title = d0.get("title") or _osp.basename(d0.get("source", ""))
                    title = _raw_title[:15] + ("…" if len(_raw_title) > 15 else "")
                    sec = d0.get("retry_in_seconds")
                    if isinstance(sec, int) and sec >= 0:
                        if sec > 60:
                            eta = f"{sec // 60}m{sec % 60:02d}s"
                        else:
                            eta = f"{sec}s"
                    else:
                        eta = "稍後"
                    wait_cnt = len(deferred)
                    error_code = str(d0.get("error_code") or "").strip().upper()
                    last_error = str(d0.get("last_error") or "").strip().lower()
                    if error_code in ("RATE_LIMIT", "RATE_LIMIT_EXHAUSTED") or "rate limit" in last_error or "速率限制" in last_error:
                        prefix = "速率限制重試"
                    elif "post_verify" in last_error or "vdb_chunks 為空" in last_error:
                        prefix = "驗證失敗重試"
                    else:
                        prefix = "匯入重試"
                    msg = f"{prefix} {title} ~{eta}"
                    if wait_cnt > 1:
                        msg += f"  +{wait_cnt - 1}件"
                    try:
                        menu = self.query_one("#menu-bar", MenuBar)
                        menu.show_progress(msg)
                    except Exception:
                        pass
                elif not completed and not failed:
                    # No active jobs and no new events — clear progress if showing
                    try:
                        menu = self.query_one("#menu-bar", MenuBar)
                        if menu._spinner_active:
                            menu.clear_notification()
                    except Exception:
                        pass
        except Exception as e:
            self.log.warning(f"Ingestion poll error: {e}")

    def action_quit(self) -> None:
        """Override quit to stop ingestion worker before exit."""
        try:
            from rag.ingestion_service import get_ingestion_service
            svc = get_ingestion_service()
            svc.stop_worker()
        except Exception:
            pass
        if hasattr(self, "_ingestion_poll_timer"):
            self._ingestion_poll_timer.stop()
        self.exit()

    async def _check_embedding_model_ready(self) -> None:
        """Compatibility diagnostics hook.

        Keeps legacy tests green; does not trigger installation/download.
        """
        try:
            from embeddings.service import get_embedding_service
            svc = get_embedding_service()
            await asyncio.to_thread(svc.get_device_diagnostics)
        except Exception:
            pass
