"""De-insight 設定 — 統一流程：選 Provider → 設定 → 選模型"""

from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

ENV_PATH = Path(__file__).resolve().parent / ".env"

PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"],
        "key_env": "OPENAI_API_KEY",
        "base_env": "OPENAI_API_BASE",
        "model_prefix": "openai/",
        "default_base": "",
        "auth_type": "api_key",
    },
    "codex": {
        "label": "OpenAI Codex (API)",
        "models": ["codex-mini-latest", "gpt-5.3-codex-medium"],
        "key_env": "CODEX_API_KEY",
        "base_env": "",
        "model_prefix": "codex/",
        "default_base": "",
        "auth_type": "api_key",
    },
    "codex-cli": {
        "label": "Codex CLI (OAuth)",
        "models": [],  # auto-populated from ~/.codex/models_cache.json
        "key_env": "",
        "base_env": "",
        "model_prefix": "codex-cli/",
        "default_base": "",
        "auth_type": "oauth",
    },
    "anthropic": {
        "label": "Anthropic",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
        ],
        "key_env": "ANTHROPIC_API_KEY",
        "base_env": "",
        "model_prefix": "",
        "default_base": "",
        "auth_type": "api_key",
    },
    "minimax": {
        "label": "MiniMax",
        "models": ["MiniMax-M2.5"],
        "key_env": "MINIMAX_API_KEY",
        "base_env": "MINIMAX_API_BASE",
        "model_prefix": "openai/",
        "default_base": "https://api.minimaxi.chat/v1",
        "auth_type": "api_key",
    },
    "ollama": {
        "label": "Ollama (本地)",
        "models": ["llama3.2", "qwen2.5", "mistral", "deepseek-r1"],
        "key_env": "",
        "base_env": "",
        "model_prefix": "ollama/",
        "default_base": "",
        "auth_type": "none",
    },
}


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


def _full_model_id(provider_id: str, model_name: str) -> str:
    return PROVIDERS[provider_id]["model_prefix"] + model_name


def _provider_connected(pid: str, env: dict[str, str]) -> bool:
    pinfo = PROVIDERS[pid]
    if pinfo["auth_type"] == "none":
        return True
    if pinfo["auth_type"] == "oauth":
        return True
    key_env = pinfo["key_env"]
    return bool(key_env and env.get(key_env))


def _mask_key(key: str) -> str:
    if not key:
        return "(未設定)"
    if len(key) <= 8:
        return "••••••••"
    return key[:4] + "••••" + key[-4:]


def _active_provider_env(pid: str, env: dict[str, str]) -> dict[str, str]:
    """Build the env vars needed for runtime, mapping provider-specific keys
    to the generic OPENAI_API_KEY / OPENAI_API_BASE that LiteLLM expects."""
    out = dict(env)
    pinfo = PROVIDERS[pid]

    # Map provider key → OPENAI_API_KEY (for openai-compatible providers)
    if pinfo["key_env"] and pinfo["key_env"] != "OPENAI_API_KEY":
        val = env.get(pinfo["key_env"], "")
        if val and pinfo["model_prefix"] in ("openai/", "codex/"):
            out["OPENAI_API_KEY"] = val

    # Map provider base → OPENAI_API_BASE
    if pinfo["base_env"] and pinfo["base_env"] != "OPENAI_API_BASE":
        val = env.get(pinfo["base_env"], "") or pinfo["default_base"]
        if val:
            out["OPENAI_API_BASE"] = val
        elif "OPENAI_API_BASE" in out:
            del out["OPENAI_API_BASE"]

    # Provider uses standard OPENAI_API_KEY directly
    if pinfo["key_env"] == "OPENAI_API_KEY":
        pass  # already in env

    # Clear OPENAI_API_BASE if provider doesn't need it
    if not pinfo["base_env"] and not pinfo["default_base"]:
        out.pop("OPENAI_API_BASE", None)

    return out


# ══════════════════════════════════════════════════════════════════════
#  SettingsScreen — 統一流程：Provider → 設定 → 模型
# ══════════════════════════════════════════════════════════════════════


class SettingsScreen(ModalScreen[str | None]):
    """選 Provider → 設定連線 → 選模型，一個介面完成。"""

    BINDINGS = [Binding("escape", "go_back", "返回")]

    # step: "provider" | "setup" | "model"

    CSS = """
    SettingsScreen {
        align: center middle;
    }
    #box {
        width: 56;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        border: round #3a3a3a;
        border-title-color: #fafafa;
        background: #0a0a0a;
    }
    #scroll {
        height: auto;
        max-height: 100%;
    }
    .sep {
        height: 1; margin: 0; color: #2a2a2a;
    }
    .hint {
        height: auto; margin: 0; padding: 0; color: #484f58;
    }
    .step-title {
        height: 1; margin: 0; padding: 0; color: #8b949e;
    }
    /* ── provider list ── */
    .prov-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .prov-btn:hover {
        background: #1a1a1a; color: #fafafa;
    }
    .prov-btn.-connected {
        color: #c9d1d9;
    }
    /* ── setup section ── */
    #setup-section {
        height: auto; margin: 0; padding: 0;
    }
    #setup-key-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #setup-key-input:focus {
        border: tall #fafafa 40%;
    }
    #setup-base-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #setup-base-input:focus {
        border: tall #fafafa 40%;
    }
    #setup-base-section {
        height: auto; margin: 0; padding: 0;
    }
    #btn-save-key {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #btn-skip-setup {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 16;
    }
    #oauth-section {
        height: auto; margin: 0; padding: 0;
    }
    #oauth-status {
        height: 1; margin: 0; padding: 0; color: #6e7681;
    }
    #btn-oauth {
        background: #2a2a2a; color: #fafafa; border: none;
        margin: 0; min-width: 20;
    }
    #btn-oauth-continue {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    /* ── model list ── */
    .model-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .model-btn:hover {
        background: #1a1a1a; color: #fafafa;
    }
    .model-btn.-active {
        color: #fafafa; text-style: bold;
    }
    /* ── back button ── */
    #btn-back {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    .btn-row {
        height: auto; margin: 1 0 0 0;
    }
    .btn-row Button {
        margin: 0 1 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._env = load_env()
        self._current_model = self._env.get("LLM_MODEL", "")
        self._step = "provider"
        self._selected_pid: str | None = None

    def compose(self) -> ComposeResult:
        box = Vertical(id="box")
        box.border_title = "⚙ Settings"
        with box:
            with VerticalScroll(id="scroll"):
                # All three steps are mounted; visibility toggled
                # ── Step 1: Provider ──
                with Vertical(id="step-provider"):
                    yield Static("◇ 選擇 Provider", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]────────────────────────────────────────[/]",
                        classes="sep",
                    )
                    for pid, pinfo in PROVIDERS.items():
                        connected = _provider_connected(pid, self._env)
                        icon = "✓" if connected else "○"
                        auth_note = ""
                        if pinfo["auth_type"] == "oauth":
                            auth_note = "  [dim]OAuth[/]"
                        elif pinfo["auth_type"] == "none":
                            auth_note = "  [dim]本地[/]"
                        yield Button(
                            f"{icon} {pinfo['label']}{auth_note}",
                            id=f"prov-{pid}",
                            classes="prov-btn" + (" -connected" if connected else ""),
                            name=pid,
                        )

                # ── Step 2: Setup (API key / OAuth) ──
                with Vertical(id="step-setup"):
                    yield Static("", id="setup-title", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]────────────────────────────────────────[/]",
                        classes="sep",
                    )
                    # API key sub-section
                    with Vertical(id="setup-section"):
                        yield Static("", id="setup-hint", classes="hint")
                        yield Input(
                            placeholder="貼上 API Key",
                            id="setup-key-input",
                            password=True,
                        )
                        with Vertical(id="setup-base-section"):
                            yield Label(
                                "◇ API Base URL", classes="step-title"
                            )
                            yield Input(
                                placeholder="自訂 base URL（留空使用預設）",
                                id="setup-base-input",
                            )
                        with Vertical(classes="btn-row"):
                            yield Button("◆ 儲存並選模型", id="btn-save-key")

                    # OAuth sub-section
                    with Vertical(id="oauth-section"):
                        yield Static("", id="oauth-status")
                        yield Button(
                            "◈ 開啟瀏覽器登入", id="btn-oauth"
                        )
                        yield Button(
                            "◆ 繼續選模型", id="btn-oauth-continue"
                        )

                    with Vertical(classes="btn-row"):
                        yield Button("◁ 返回", id="btn-back")

                # ── Step 3: Model ──
                with Vertical(id="step-model"):
                    yield Static("", id="model-title", classes="step-title")
                    yield Static(
                        "[dim #2a2a2a]────────────────────────────────────────[/]",
                        classes="sep",
                    )
                    yield Vertical(id="model-list")
                    with Vertical(classes="btn-row"):
                        yield Button("◁ 返回", id="btn-back-model")

    async def on_mount(self) -> None:
        self._show_step("provider")

    # ── step transitions ──

    def _show_step(self, step: str) -> None:
        self._step = step
        self.query_one("#step-provider").display = step == "provider"
        self.query_one("#step-setup").display = step == "setup"
        self.query_one("#step-model").display = step == "model"

        box = self.query_one("#box", Vertical)
        if step == "provider":
            box.border_title = "⚙ Settings"
        elif step == "setup":
            pid = self._selected_pid or ""
            label = PROVIDERS.get(pid, {}).get("label", "")
            box.border_title = f"⚙ {label} — 連線設定"
        elif step == "model":
            pid = self._selected_pid or ""
            label = PROVIDERS.get(pid, {}).get("label", "")
            box.border_title = f"⚙ {label} — 選擇模型"

    def _goto_provider(self) -> None:
        self._show_step("provider")

    def _goto_setup(self, pid: str) -> None:
        self._selected_pid = pid
        pinfo = PROVIDERS[pid]

        if pinfo["auth_type"] == "api_key":
            # Show API key input
            self.query_one("#setup-section").display = True
            self.query_one("#oauth-section").display = False

            key_env = pinfo["key_env"]
            current = self._env.get(key_env, "")
            masked = _mask_key(current)
            self.query_one("#setup-title", Static).update(
                f"◇ {pinfo['label']} — API Key"
            )
            self.query_one("#setup-hint", Static).update(
                f"[dim #484f58]環境變數: {key_env}\n目前: {masked}[/]"
            )
            self.query_one("#setup-key-input", Input).value = ""

            # Base URL
            if pinfo["base_env"]:
                self.query_one("#setup-base-section").display = True
                cur_base = self._env.get(pinfo["base_env"], "") or pinfo["default_base"]
                self.query_one("#setup-base-input", Input).value = cur_base
            else:
                self.query_one("#setup-base-section").display = False

        elif pinfo["auth_type"] == "oauth":
            self.query_one("#setup-section").display = False
            self.query_one("#oauth-section").display = True
            self.query_one("#setup-title", Static).update(
                f"◈ {pinfo['label']} — OAuth 登入"
            )
            self.query_one("#oauth-status", Static).update(
                "[dim #484f58]檢查登入狀態…[/]"
            )
            self._check_oauth_status()

        elif pinfo["auth_type"] == "none":
            # No setup needed, go straight to model
            self._goto_model(pid)
            return

        self._show_step("setup")
        # Focus key input if visible
        if pinfo["auth_type"] == "api_key":
            self.query_one("#setup-key-input", Input).focus()

    def _goto_model(self, pid: str) -> None:
        self._selected_pid = pid
        pinfo = PROVIDERS[pid]

        # Codex CLI: auto-read models from cache
        models = list(pinfo["models"])
        if pid == "codex-cli":
            from codex_client import get_codex_models
            cached = get_codex_models()
            if cached:
                models = cached

        self._pending_models = models
        self.query_one("#model-title", Static).update(
            f"◇ {pinfo['label']} — 選擇模型"
        )

        self._show_step("model")
        self.call_after_refresh(self._populate_models)

    async def _populate_models(self) -> None:
        pid = self._selected_pid
        if not pid:
            return
        models = getattr(self, "_pending_models", [])

        model_list = self.query_one("#model-list", Vertical)
        await model_list.remove_children()
        for idx, m in enumerate(models):
            full_id = _full_model_id(pid, m)
            is_active = full_id == self._current_model
            btn = Button(
                f"{'◆' if is_active else '◇'} {m}",
                id=f"mdl-{pid}-{idx}",
                classes="model-btn" + (" -active" if is_active else ""),
                name=f"{pid}|{m}",
            )
            await model_list.mount(btn)

    # ── button handling ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        # Provider selection
        if bid.startswith("prov-"):
            pid = event.button.name or ""
            if pid not in PROVIDERS:
                return
            # If already connected, skip setup → go to models
            if _provider_connected(pid, self._env):
                self._goto_model(pid)
            else:
                self._goto_setup(pid)

        # Save API key → go to models
        elif bid == "btn-save-key":
            self._save_key_and_continue()

        # OAuth buttons
        elif bid == "btn-oauth":
            self._do_oauth_login()
        elif bid == "btn-oauth-continue":
            if self._selected_pid:
                self._goto_model(self._selected_pid)

        # Back buttons
        elif bid in ("btn-back", "btn-back-model"):
            if self._step == "model":
                # Back to provider (not setup, since already connected)
                self._goto_provider()
            elif self._step == "setup":
                self._goto_provider()

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
        pinfo = PROVIDERS[pid]

        new_key = self.query_one("#setup-key-input", Input).value.strip()
        if not new_key:
            self.notify("請輸入 API Key")
            return

        # Save key
        self._env[pinfo["key_env"]] = new_key

        # Save base URL
        if pinfo["base_env"]:
            new_base = self.query_one("#setup-base-input", Input).value.strip()
            if new_base:
                self._env[pinfo["base_env"]] = new_base
            elif pinfo["default_base"]:
                self._env[pinfo["base_env"]] = pinfo["default_base"]

        save_env(self._env)
        self.notify(f"✓ {pinfo['label']} 已連線")

        # Update provider button
        try:
            btn = self.query_one(f"#prov-{pid}", Button)
            auth_note = ""
            btn.label = f"✓ {pinfo['label']}{auth_note}"
            btn.add_class("-connected")
        except Exception:
            pass

        # Proceed to model selection
        self._goto_model(pid)

    def _select_model(self, pid: str, model_name: str) -> None:
        """選定模型，儲存 .env，關閉畫面。"""
        pinfo = PROVIDERS[pid]
        full_id = _full_model_id(pid, model_name)

        # Keep all existing provider keys, just update model + runtime vars
        env = dict(self._env)
        env["LLM_MODEL"] = full_id

        # Map provider-specific key → OPENAI_API_KEY for LiteLLM
        if pinfo["key_env"]:
            key_val = env.get(pinfo["key_env"], "")
            if key_val and pinfo["key_env"] != "ANTHROPIC_API_KEY":
                env["OPENAI_API_KEY"] = key_val

        # Map provider-specific base → OPENAI_API_BASE
        if pinfo["default_base"]:
            base_val = env.get(pinfo.get("base_env", ""), "") or pinfo["default_base"]
            env["OPENAI_API_BASE"] = base_val
        elif not pinfo["base_env"]:
            env.pop("OPENAI_API_BASE", None)

        save_env(env)
        self.dismiss(full_id)

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
                "[#8b949e]✗ codex CLI 未安裝[/]  [dim]npm i -g @openai/codex[/]",
            )
            return

        status = await codex_login_status()
        if "logged in" in status.lower() or "chatgpt" in status.lower():
            self.app.call_from_thread(
                status_w.update, f"[#8b949e]✓ {status}[/]"
            )
        else:
            self.app.call_from_thread(
                status_w.update, f"[#8b949e]○ {status}[/]"
            )

    @work(thread=True)
    async def _do_oauth_login(self) -> None:
        from codex_client import codex_login

        try:
            status_w = self.query_one("#oauth-status", Static)
            self.app.call_from_thread(
                status_w.update, "[italic #8b949e]⟳ 正在開啟瀏覽器…[/]"
            )
        except Exception:
            pass

        success, msg = await codex_login()

        try:
            status_w = self.query_one("#oauth-status", Static)
            if success:
                self.app.call_from_thread(
                    status_w.update, f"[#8b949e]✓ {msg}[/]"
                )
            else:
                self.app.call_from_thread(
                    status_w.update, f"[#8b949e]✗ {msg}[/]"
                )
        except Exception:
            pass

    # ── navigation ──

    def action_go_back(self) -> None:
        if self._step == "provider":
            self.dismiss(None)
        elif self._step in ("setup", "model"):
            self._goto_provider()
