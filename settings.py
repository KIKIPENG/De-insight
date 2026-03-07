"""De-insight 設定 — Provider 模式管理所有 API 服務"""

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from providers import (
    SERVICES, CHAT_PROVIDERS, EMBED_PROVIDERS, RAG_LLM_PROVIDERS, PROVIDERS,
)

ENV_PATH = Path(__file__).resolve().parent / ".env"


# ── ENV helpers ──────────────────────────────────────────────────────


def load_env() -> dict[str, str]:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def save_env(env: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in env.items() if v]
    ENV_PATH.write_text("\n".join(lines) + "\n")


def save_env_key(key: str, value: str) -> None:
    """把單個 key 寫入 .env 檔案。若 .env 不存在則建立。"""
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    found = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n")


def _full_model_id(provider_id: str, model_name: str, providers: dict = None) -> str:
    providers = providers or CHAT_PROVIDERS
    prefix = providers.get(provider_id, {}).get("model_prefix", "")
    return prefix + model_name


def _provider_connected(pid: str, env: dict[str, str], providers: dict = None) -> bool:
    providers = providers or CHAT_PROVIDERS
    pinfo = providers.get(pid, {})
    if pinfo.get("auth_type") in ("none", "oauth"):
        return True
    key_env = pinfo.get("key_env", "")
    return bool(key_env and env.get(key_env))


def _mask_key(key: str) -> str:
    if not key:
        return "(未設定)"
    if len(key) <= 8:
        return "--------"
    return key[:4] + "----" + key[-4:]


def _get_service_status(env: dict) -> dict[str, str]:
    """回傳每個服務目前的設定狀態摘要。"""
    status = {}
    # Chat
    chat_model = env.get("LLM_MODEL", "")
    status["chat"] = chat_model if chat_model else "(未設定)"
    # Embedding
    embed_provider = env.get("EMBED_PROVIDER", "")
    embed_model = env.get("EMBED_MODEL", "")
    if embed_provider and embed_model:
        status["embedding"] = f"{embed_provider}/{embed_model}"
    elif env.get("JINA_API_KEY"):
        status["embedding"] = "jina/jina-embeddings-v3"
    else:
        status["embedding"] = "(未設定)"
    # RAG LLM
    rag_model = env.get("RAG_LLM_MODEL", "")
    status["rag_llm"] = rag_model if rag_model else "(跟聊天模型相同)"
    return status


# ══════════════════════════════════════════════════════════════════════
#  SettingsScreen
# ══════════════════════════════════════════════════════════════════════


class SettingsScreen(ModalScreen[str | None]):
    """統一設定：服務總覽 → Provider → 設定 → 模型。"""

    BINDINGS = [Binding("escape", "go_back", "返回")]

    CSS = """
    SettingsScreen { align: center middle; }
    #box {
        width: 58; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; border-title-color: #fafafa;
        background: #0a0a0a;
    }
    #scroll { height: auto; max-height: 100%; }
    .sep { height: 1; margin: 0; color: #2a2a2a; }
    .hint { height: auto; margin: 0; padding: 0; color: #484f58; }
    .step-title { height: 1; margin: 0; padding: 0; color: #8b949e; }
    /* service overview */
    .svc-btn {
        background: transparent; color: #8b949e;
        border: none; height: auto; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .svc-btn:hover { background: #1a1a1a; color: #fafafa; }
    .svc-btn.-configured { color: #c9d1d9; }
    .svc-status { height: auto; margin: 0; padding: 0 2; color: #484f58; }
    /* provider list */
    .prov-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .prov-btn:hover { background: #1a1a1a; color: #fafafa; }
    .prov-btn.-connected { color: #c9d1d9; }
    /* setup section */
    #setup-section { height: auto; margin: 0; padding: 0; }
    #setup-key-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #setup-key-input:focus { border: tall #fafafa 40%; }
    #setup-base-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #setup-base-input:focus { border: tall #fafafa 40%; }
    #setup-base-section { height: auto; margin: 0; padding: 0; }
    #btn-save-key {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #oauth-section { height: auto; margin: 0; padding: 0; }
    #oauth-status { height: 1; margin: 0; padding: 0; color: #6e7681; }
    #btn-oauth {
        background: #2a2a2a; color: #fafafa; border: none;
        margin: 0; min-width: 20;
    }
    #btn-oauth-continue {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    /* model list */
    .model-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .model-btn:hover { background: #1a1a1a; color: #fafafa; }
    .model-btn.-active { color: #fafafa; text-style: bold; }
    /* navigation */
    #btn-back {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    .btn-row { height: auto; margin: 1 0 0 0; }
    .btn-row Button { margin: 0 1 0 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._env = load_env()
        self._current_model = self._env.get("LLM_MODEL", "")
        # steps: "services" | "provider" | "setup" | "model"
        self._step = "services"
        self._active_service: str = "chat"  # which service we're configuring
        self._selected_pid: str | None = None

    def _get_providers(self) -> dict:
        """取得當前服務對應的 provider 清單。"""
        if self._active_service == "chat":
            return CHAT_PROVIDERS
        elif self._active_service == "embedding":
            return EMBED_PROVIDERS
        elif self._active_service == "rag_llm":
            return RAG_LLM_PROVIDERS
        return CHAT_PROVIDERS

    def compose(self) -> ComposeResult:
        box = Vertical(id="box")
        box.border_title = "Settings"
        with box:
            with VerticalScroll(id="scroll"):
                # ── Step 0: Service overview ──
                with Vertical(id="step-services"):
                    yield Static("  選擇要設定的服務", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]" + "-" * 50 + "[/]", classes="sep",
                    )
                    for sid, sinfo in SERVICES.items():
                        yield Button(
                            f"  {sinfo['label']}",
                            id=f"svc-{sid}",
                            classes="svc-btn",
                            name=sid,
                        )
                        yield Static("", id=f"svc-status-{sid}", classes="svc-status")
                    yield Static(
                        "[dim #2a2a2a]" + "-" * 50 + "[/]", classes="sep",
                    )
                    yield Button("<- 回到對話", classes="back-btn")

                # ── Step 1: Provider ──
                with Vertical(id="step-provider"):
                    yield Static("", id="prov-title", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]" + "-" * 50 + "[/]", classes="sep",
                    )
                    yield Vertical(id="provider-list")
                    with Vertical(classes="btn-row"):
                        yield Button("<- 返回", id="btn-back")

                # ── Step 2: Setup ──
                with Vertical(id="step-setup"):
                    yield Static("", id="setup-title", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]" + "-" * 50 + "[/]", classes="sep",
                    )
                    with Vertical(id="setup-section"):
                        yield Static("", id="setup-hint", classes="hint")
                        yield Input(
                            placeholder="貼上 API Key",
                            id="setup-key-input",
                            password=True,
                        )
                        with Vertical(id="setup-base-section"):
                            yield Label("  API Base URL", classes="step-title")
                            yield Input(
                                placeholder="自訂 base URL（留空使用預設）",
                                id="setup-base-input",
                            )
                        with Vertical(classes="btn-row"):
                            yield Button("  儲存並選模型", id="btn-save-key")
                    with Vertical(id="oauth-section"):
                        yield Static("", id="oauth-status")
                        yield Button("  開啟瀏覽器登入", id="btn-oauth")
                        yield Button("  繼續選模型", id="btn-oauth-continue")
                    with Vertical(classes="btn-row"):
                        yield Button("<- 返回", id="btn-back-setup")

                # ── Step 3: Model ──
                with Vertical(id="step-model"):
                    yield Static("", id="model-title", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]" + "-" * 50 + "[/]", classes="sep",
                    )
                    yield Vertical(id="model-list")
                    with Vertical(classes="btn-row"):
                        yield Button("<- 返回", id="btn-back-model")

    async def on_mount(self) -> None:
        self._show_step("services")
        self._refresh_service_status()

    # ── step transitions ──

    def _show_step(self, step: str) -> None:
        self._step = step
        self.query_one("#step-services").display = step == "services"
        self.query_one("#step-provider").display = step == "provider"
        self.query_one("#step-setup").display = step == "setup"
        self.query_one("#step-model").display = step == "model"

        box = self.query_one("#box", Vertical)
        if step == "services":
            box.border_title = "Settings"
        elif step == "provider":
            svc = SERVICES.get(self._active_service, {})
            box.border_title = f"Settings > {svc.get('label', '')}"
        elif step == "setup":
            pid = self._selected_pid or ""
            providers = self._get_providers()
            label = providers.get(pid, {}).get("label", "")
            box.border_title = f"Settings > {label} -- 連線設定"
        elif step == "model":
            pid = self._selected_pid or ""
            providers = self._get_providers()
            label = providers.get(pid, {}).get("label", "")
            box.border_title = f"Settings > {label} -- 選擇模型"

    def _refresh_service_status(self) -> None:
        statuses = _get_service_status(self._env)
        for sid, status_text in statuses.items():
            try:
                w = self.query_one(f"#svc-status-{sid}", Static)
                configured = status_text != "(未設定)"
                w.update(f"    [dim]{status_text}[/]")
                btn = self.query_one(f"#svc-{sid}", Button)
                if configured:
                    btn.add_class("-configured")
                else:
                    btn.remove_class("-configured")
            except Exception:
                pass

    def _goto_services(self) -> None:
        self._refresh_service_status()
        self._show_step("services")

    async def _goto_provider(self, service: str) -> None:
        self._active_service = service
        providers = self._get_providers()
        svc = SERVICES.get(service, {})

        self.query_one("#prov-title", Static).update(
            f"  {svc.get('label', '')} — 選擇 Provider"
        )

        plist = self.query_one("#provider-list", Vertical)
        await plist.remove_children()
        for pid, pinfo in providers.items():
            connected = _provider_connected(pid, self._env, providers)
            icon = "+" if connected else "o"
            auth_note = ""
            if pinfo.get("auth_type") == "oauth":
                auth_note = "  [dim]OAuth[/]"
            elif pinfo.get("auth_type") == "none":
                auth_note = "  [dim]本地[/]"
            await plist.mount(Button(
                f"{icon} {pinfo['label']}{auth_note}",
                id=f"prov-{pid}",
                classes="prov-btn" + (" -connected" if connected else ""),
                name=pid,
            ))

        self._show_step("provider")

    def _goto_setup(self, pid: str) -> None:
        self._selected_pid = pid
        providers = self._get_providers()
        pinfo = providers[pid]

        if pinfo.get("auth_type") == "api_key":
            self.query_one("#setup-section").display = True
            self.query_one("#oauth-section").display = False

            key_env = pinfo["key_env"]
            current = self._env.get(key_env, "")
            masked = _mask_key(current)
            self.query_one("#setup-title", Static).update(
                f"  {pinfo['label']} — API Key"
            )
            hint = f"[dim #484f58]環境變數: {key_env}\n目前: {masked}[/]"
            if current:
                hint += "\n[dim #484f58]留空直接按儲存 = 保留目前的 key[/]"
            self.query_one("#setup-hint", Static).update(hint)
            self.query_one("#setup-key-input", Input).value = ""

            if pinfo.get("base_env"):
                self.query_one("#setup-base-section").display = True
                cur_base = self._env.get(pinfo["base_env"], "") or pinfo.get("default_base", "")
                self.query_one("#setup-base-input", Input).value = cur_base
            else:
                self.query_one("#setup-base-section").display = False

        elif pinfo.get("auth_type") == "oauth":
            self.query_one("#setup-section").display = False
            self.query_one("#oauth-section").display = True
            self.query_one("#setup-title", Static).update(
                f"  {pinfo['label']} — OAuth 登入"
            )
            self.query_one("#oauth-status", Static).update(
                "[dim #484f58]檢查登入狀態...[/]"
            )
            self._check_oauth_status()

        elif pinfo.get("auth_type") == "none":
            self._goto_model(pid)
            return

        self._show_step("setup")
        if pinfo.get("auth_type") == "api_key":
            self.query_one("#setup-key-input", Input).focus()

    def _goto_model(self, pid: str) -> None:
        self._selected_pid = pid
        providers = self._get_providers()
        pinfo = providers[pid]

        # Special: "same-as-chat"
        if pid == "same-as-chat":
            self._env.pop("RAG_LLM_MODEL", None)
            self._env.pop("RAG_API_KEY", None)
            self._env.pop("RAG_API_BASE", None)
            save_env(self._env)
            self.notify("知識庫 LLM 已設為跟聊天模型相同")
            self._goto_services()
            return

        models = list(pinfo.get("models", []))
        if pid == "codex-cli":
            from codex_client import get_codex_models
            cached = get_codex_models()
            if cached:
                models = cached

        self._pending_models = models
        self.query_one("#model-title", Static).update(
            f"  {pinfo['label']} — 選擇模型"
        )

        self._show_step("model")
        self.call_after_refresh(self._populate_models)

    async def _populate_models(self) -> None:
        pid = self._selected_pid
        if not pid:
            return
        models = getattr(self, "_pending_models", [])
        providers = self._get_providers()

        # Determine current model for this service
        if self._active_service == "chat":
            current = self._env.get("LLM_MODEL", "")
        elif self._active_service == "embedding":
            current = self._env.get("EMBED_MODEL", "")
        elif self._active_service == "rag_llm":
            current = self._env.get("RAG_LLM_MODEL", "")
        else:
            current = ""

        model_list = self.query_one("#model-list", Vertical)
        await model_list.remove_children()
        for idx, m in enumerate(models):
            full_id = _full_model_id(pid, m, providers)
            is_active = full_id == current or m == current
            btn = Button(
                f"{'*' if is_active else 'o'} {m}",
                id=f"mdl-{pid}-{idx}",
                classes="model-btn" + (" -active" if is_active else ""),
                name=f"{pid}|{m}",
            )
            await model_list.mount(btn)

    # ── button handling ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        # Back to chat
        if event.button.has_class("back-btn"):
            self.dismiss(None)
            return

        # Service selection
        if bid.startswith("svc-"):
            service = event.button.name or ""
            if service in SERVICES:
                self.call_after_refresh(lambda: self._goto_provider(service))

        # Provider selection
        elif bid.startswith("prov-"):
            pid = event.button.name or ""
            providers = self._get_providers()
            if pid not in providers:
                return
            pinfo = providers[pid]
            if pinfo.get("auth_type") in ("none", "oauth"):
                # 不需要 key 的 provider，直接進設定/模型
                if pinfo.get("auth_type") == "none":
                    self._goto_model(pid)
                else:
                    self._goto_setup(pid)
            else:
                # 有 API key 的 provider，一律進設定頁（可重新輸入 key）
                self._goto_setup(pid)

        # Save API key
        elif bid == "btn-save-key":
            self._save_key_and_continue()

        # OAuth
        elif bid == "btn-oauth":
            self._do_oauth_login()
        elif bid == "btn-oauth-continue":
            if self._selected_pid:
                self._goto_model(self._selected_pid)

        # Back buttons
        elif bid == "btn-back":
            self._goto_services()
        elif bid == "btn-back-setup":
            self.call_after_refresh(
                lambda: self._goto_provider(self._active_service)
            )
        elif bid == "btn-back-model":
            self.call_after_refresh(
                lambda: self._goto_provider(self._active_service)
            )

        # Model selection
        elif bid.startswith("mdl-"):
            name = event.button.name or ""
            if "|" in name:
                pid, model_name = name.split("|", 1)
                self._select_model(pid, model_name)

    def _save_key_and_continue(self) -> None:
        pid = self._selected_pid
        if not pid:
            return
        providers = self._get_providers()
        pinfo = providers[pid]

        new_key = self.query_one("#setup-key-input", Input).value.strip()
        existing_key = self._env.get(pinfo["key_env"], "")
        if not new_key and not existing_key:
            self.notify("請輸入 API Key")
            return

        if new_key:
            self._env[pinfo["key_env"]] = new_key

        if pinfo.get("base_env"):
            new_base = self.query_one("#setup-base-input", Input).value.strip()
            if new_base:
                self._env[pinfo["base_env"]] = new_base
            elif pinfo.get("default_base"):
                self._env[pinfo["base_env"]] = pinfo["default_base"]

        save_env(self._env)
        self.notify(f"+ {pinfo['label']} 已連線")

        self._goto_model(pid)

    def _select_model(self, pid: str, model_name: str) -> None:
        """選定模型，儲存 .env。"""
        providers = self._get_providers()
        pinfo = providers[pid]
        full_id = _full_model_id(pid, model_name, providers)

        if self._active_service == "chat":
            self._save_chat_model(pid, model_name, full_id, pinfo)
        elif self._active_service == "embedding":
            self._save_embed_model(pid, model_name, pinfo)
        elif self._active_service == "rag_llm":
            self._save_rag_model(pid, model_name, full_id, pinfo)

        save_env(self._env)
        self.notify(f"已設定: {model_name}")
        self._goto_services()

    def _save_chat_model(self, pid, model_name, full_id, pinfo):
        self._env["LLM_MODEL"] = full_id

        # 1. 更新 API Base
        default_base = pinfo.get("default_base", "")
        base_env = pinfo.get("base_env", "")
        if default_base:
            base_val = (self._env.get(base_env, "") if base_env else "") or default_base
            self._env["OPENAI_API_BASE"] = base_val
        else:
            self._env.pop("OPENAI_API_BASE", None)

        # 2. FastAPI backend 相容層：把目前 provider 的 key 鏡像到 OPENAI_API_KEY
        key_env = pinfo.get("key_env", "")
        if key_env and key_env != "ANTHROPIC_API_KEY":
            current_key = self._env.get(key_env, "")
            if current_key:
                self._env["OPENAI_API_KEY"] = current_key
        elif not key_env:
            self._env.pop("OPENAI_API_KEY", None)

    def _save_embed_model(self, pid, model_name, pinfo):
        self._env["EMBED_PROVIDER"] = pid
        self._env["EMBED_MODEL"] = model_name
        dims = pinfo.get("dims", {})
        self._env["EMBED_DIM"] = str(dims.get(model_name, 1024))
        if pinfo.get("key_env"):
            key_val = self._env.get(pinfo["key_env"], "")
            if key_val:
                self._env["EMBED_API_KEY"] = key_val
        self._env["EMBED_API_BASE"] = pinfo.get("default_base", "")
        # Backward compat for Jina
        if pid == "jina" and pinfo.get("key_env"):
            self._env["JINA_API_KEY"] = self._env.get(pinfo["key_env"], "")

    def _save_rag_model(self, pid, model_name, full_id, pinfo):
        self._env["RAG_LLM_MODEL"] = full_id
        if pinfo.get("key_env"):
            self._env["RAG_API_KEY"] = self._env.get(pinfo["key_env"], "")
        base = pinfo.get("default_base", "")
        if base:
            self._env["RAG_API_BASE"] = base
        else:
            self._env.pop("RAG_API_BASE", None)

    # ── OAuth ──

    @work(thread=True)
    async def _check_oauth_status(self) -> None:
        from codex_client import codex_login_status, is_codex_available

        try:
            status_w = self.query_one("#oauth-status", Static)
        except Exception:
            return

        if not is_codex_available():
            self.app.call_from_thread(
                status_w.update,
                "[#8b949e]x codex CLI 未安裝[/]  [dim]npm i -g @openai/codex[/]",
            )
            return

        status = await codex_login_status()
        if "logged in" in status.lower() or "chatgpt" in status.lower():
            self.app.call_from_thread(
                status_w.update, f"[#8b949e]+ {status}[/]"
            )
        else:
            self.app.call_from_thread(
                status_w.update, f"[#8b949e]o {status}[/]"
            )

    @work(thread=True)
    async def _do_oauth_login(self) -> None:
        from codex_client import codex_login

        try:
            status_w = self.query_one("#oauth-status", Static)
            self.app.call_from_thread(
                status_w.update, "[italic #8b949e]正在開啟瀏覽器...[/]"
            )
        except Exception:
            pass

        success, msg = await codex_login()

        try:
            status_w = self.query_one("#oauth-status", Static)
            if success:
                self.app.call_from_thread(
                    status_w.update, f"[#8b949e]+ {msg}[/]"
                )
            else:
                self.app.call_from_thread(
                    status_w.update, f"[#8b949e]x {msg}[/]"
                )
        except Exception:
            pass

    # ── navigation ──

    def action_go_back(self) -> None:
        if self._step == "services":
            self.dismiss(None)
        elif self._step == "provider":
            self._goto_services()
        elif self._step == "setup":
            self.call_after_refresh(
                lambda: self._goto_provider(self._active_service)
            )
        elif self._step == "model":
            self.call_after_refresh(
                lambda: self._goto_provider(self._active_service)
            )
