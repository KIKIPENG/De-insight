"""v0.7 Modals — all ModalScreen subclasses live here."""

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, ProgressBar, Static, TextArea

from memory.store import delete_memory, get_memories, get_memory_stats


def _get_project_paths(app) -> "tuple[Path | None, Path | None]":
    """回傳 (memories_db_path, lancedb_dir)。"""
    from pathlib import Path
    from paths import project_root
    state = getattr(app, 'state', None)
    pid = state.current_project["id"] if state and state.current_project else None
    if not pid:
        return None, None
    root = project_root(pid)
    return root / "memories.db", root / "lancedb"



class OnboardingScreen(ModalScreen[str | None]):
    """首次啟動三步驟引導：Chat Provider → Embedding → 完成。"""

    CSS = """
    OnboardingScreen { align: center middle; }
    #ob-box {
        width: 62; height: auto; max-height: 85%; padding: 1 2;
        border: round #3a3a3a; background: #0a0a0a;
        border-title-color: #d4a27a;
    }
    #ob-scroll { height: auto; max-height: 100%; }
    .ob-sep { height: 1; margin: 0; color: #2a2a2a; }
    .ob-title { height: 1; margin: 0; padding: 0; color: #d4a27a; text-style: bold; }
    .ob-hint { height: auto; margin: 0; padding: 0 1; color: #484f58; }
    .ob-prov-btn {
        background: transparent; color: #8b949e;
        border: none; height: 1; margin: 0; padding: 0 2;
        min-width: 0; width: 100%;
        text-align: left; content-align: left middle;
    }
    .ob-prov-btn:hover { background: #1a1a1a; color: #fafafa; }
    #ob-key-input {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #ob-key-input:focus { border: tall #fafafa 40%; }
    #ob-model-search {
        margin: 0; background: #111111; color: #fafafa;
        border: tall #3a3a3a; padding: 0 1;
    }
    #ob-model-search:focus { border: tall #fafafa 40%; }
    #ob-model-scroll { height: 16; max-height: 45%; }
    #ob-key-section { height: auto; }
    .ob-btn-row { height: auto; margin: 1 0 0 0; }
    .ob-btn-row Button { margin: 0 1 0 0; }
    #btn-ob-save {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #btn-ob-skip {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    #btn-ob-next {
        background: #fafafa; color: #0a0a0a; border: none;
        margin: 0; min-width: 16;
    }
    #btn-ob-back {
        background: #2a2a2a; color: #8b949e; border: none;
        margin: 0; min-width: 12;
    }
    #ob-download-status { height: auto; margin: 0; padding: 0 1; color: #6e7681; }
    #ob-progress-bar { margin: 0 2; height: 1; }
    #ob-progress-bar Bar { color: #d4a27a; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._step = "chat_provider"
        self._selected_chat_pid: str | None = None
        self._selected_embed_pid: str | None = None
        self._selected_rag_pid: str | None = None
        self._selected_vision_pid: str | None = None
        self._model_search_query: str = ""
        self._model_candidates: list[str] = []
        self._model_button_kind: str | None = None

    def compose(self) -> ComposeResult:
        box = Vertical(id="ob-box")
        from paths import __version__ as _ver
        box.border_title = f"De-insight {_ver} Setup"
        with box:
            with VerticalScroll(id="ob-scroll"):
                yield Vertical(id="ob-content")

    async def on_mount(self) -> None:
        await self._render_step()

    async def _render_step(self) -> None:
        container = self.query_one("#ob-content", Vertical)
        await container.remove_children()

        if self._step == "chat_provider":
            await self._render_chat_provider(container)
        elif self._step == "chat_setup":
            await self._render_chat_setup(container)
        elif self._step == "embed":
            await self._render_embed(container)
        elif self._step == "embed_setup":
            await self._render_embed_setup(container)
        elif self._step == "embed_download":
            await self._render_embed_download(container)
        elif self._step == "rag_llm":
            await self._render_rag_llm(container)
        elif self._step == "rag_setup":
            await self._render_rag_setup(container)
        elif self._step == "vision_provider":
            await self._render_vision_provider(container)
        elif self._step == "vision_setup":
            await self._render_vision_setup(container)
        elif self._step == "done":
            await self._render_done(container)

    def _filter_models(self, models: list[str]) -> list[str]:
        q = self._model_search_query.strip().lower()
        if not q:
            return models
        return [m for m in models if q in m.lower()]

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "ob-model-search":
            self._model_search_query = event.value
            self.run_worker(self._refresh_model_buttons(), exclusive=True)

    async def _refresh_model_buttons(self) -> None:
        """Refresh only the model button list to avoid full-screen flicker."""
        try:
            scroll = self.query_one("#ob-model-scroll", VerticalScroll)
        except Exception:
            return

        await scroll.remove_children()
        models = self._filter_models(self._model_candidates)
        kind = self._model_button_kind
        for idx, model in enumerate(models):
            if kind == "chat":
                btn_id = f"ob-model-{idx}"
            elif kind == "rag":
                btn_id = (
                    "ob-ragmodel-"
                    + model.replace("/", "_").replace(":", "-").replace(".", "_")
                )
            elif kind == "vision":
                btn_id = f"ob-vmodel-{idx}"
            else:
                continue

            await scroll.mount(
                Button(
                    f"  {model}",
                    id=btn_id,
                    classes="ob-prov-btn",
                    name=model,
                )
            )

    async def _render_chat_provider(self, container: Vertical) -> None:
        from providers import CHAT_PROVIDERS
        await container.mount(
            Static("  Step 1/5: Chat Provider", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  選擇聊天用的 LLM 服務", classes="ob-hint"),
        )
        for pid, pinfo in CHAT_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-chat-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                )
            )

    async def _render_chat_setup(self, container: Vertical) -> None:
        from providers import CHAT_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = CHAT_PROVIDERS.get(self._selected_chat_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 1/5: {pinfo.get('label', '')} Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        # A3: auth_type in ("none", "oauth") -> hide key input
        if auth_type not in ("none", "oauth"):
            key_env = pinfo.get("key_env", "")
            await container.mount(
                Static(f"  請輸入 {key_env}", classes="ob-hint"),
                Vertical(
                    Input(placeholder="API Key", id="ob-key-input", password=True),
                    id="ob-key-section",
                ),
            )
        else:
            if auth_type == "oauth":
                label = "OAuth 登入（稍後在設定中完成）"
            else:
                label = "無需 API Key"
            await container.mount(Static(f"  {label}", classes="ob-hint"))

        # model selection
        models = await resolve_dynamic_models(
            provider_id=self._selected_chat_pid or "",
            service="chat",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "chat"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
                Static("  選擇模型", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for idx, model in enumerate(models):
                await scroll.mount(
                    Button(
                        f"  {model}",
                        id=f"ob-model-{idx}",
                        classes="ob-prov-btn",
                        name=model,
                    )
                )
        else:
            await container.mount(
                Horizontal(
                    Button("下一步 ->", id="btn-ob-next"),
                    Button("<- 返回", id="btn-ob-back"),
                    classes="ob-btn-row",
                ),
            )

    async def _render_embed(self, container: Vertical) -> None:
        from providers import EMBED_PROVIDERS
        await container.mount(
            Static("  Step 2/5: Embedding Model", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  選擇文字向量化的方式", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in EMBED_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-embed-provider-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                )
            )
        await container.mount(
            Horizontal(Button("<- 返回", id="btn-ob-back"), classes="ob-btn-row"),
        )

    async def _render_embed_setup(self, container: Vertical) -> None:
        """OpenRouter API Key 輸入步驟。"""
        from settings import load_env
        existing_key = load_env().get("OPENROUTER_API_KEY", "")
        hint = (
            "  已有 OPENROUTER_API_KEY（可直接下一步，或輸入新的）"
            if existing_key else
            "  請輸入 OpenRouter API Key"
        )
        await container.mount(
            Static("  Step 2/5: OpenRouter Embedding", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static(hint, classes="ob-hint"),
            Vertical(
                Input(
                    placeholder="sk-or-xxxx（留空則沿用現有）" if existing_key else "sk-or-xxxx",
                    id="ob-key-input",
                    password=True,
                ),
                id="ob-key-section",
            ),
            Horizontal(
                Button("<- 返回", id="btn-ob-back"),
                Button("儲存並繼續 ->", id="btn-ob-embed-save"),
                classes="ob-btn-row",
            ),
        )

    async def _render_embed_download(self, container: Vertical) -> None:
        await container.mount(
            Static("  Step 2/5: OpenRouter Embedding", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  OpenRouter embedding 不需要本地安裝。", id="ob-download-status"),
        )
        self._on_download_complete()

    @work(exclusive=True, thread=True)
    def _run_model_download(self) -> None:
        self.app.call_from_thread(self._on_download_complete)

    def _on_download_complete(self) -> None:
        self._step = "rag_llm"
        self.run_worker(self._render_step(), exclusive=True)

    def _on_download_error(self, msg: str) -> None:
        try:
            status = self.query_one("#ob-download-status", Static)
            status.update(f"  [red]設定失敗：{msg}[/]")
        except Exception:
            pass

    # ── Step 3: RAG LLM ──

    async def _render_rag_llm(self, container: Vertical) -> None:
        from providers import RAG_LLM_PROVIDERS
        await container.mount(
            Static("  Step 3/5: 知識庫建圖 LLM", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  匯入文件時需要 LLM 抽取知識圖譜", classes="ob-hint"),
            Static("  建議選低成本 API（如 Google AI Studio 免費額度充足）", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in RAG_LLM_PROVIDERS.items():
            if pid == "ollama" or pid == "ollama-local":
                continue
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-rag-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                ),
            )

    async def _render_rag_setup(self, container: Vertical) -> None:
        from providers import RAG_LLM_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = RAG_LLM_PROVIDERS.get(self._selected_rag_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 3/5: {pinfo.get('label', '')} RAG Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        key_env = pinfo.get("key_env", "")
        if auth_type == "api_key" and key_env:
            from settings import load_env
            existing_key = load_env().get(key_env, "")
            hint = f"  已有 {key_env}（可直接選模型，或輸入新的 Key）" if existing_key else f"  請輸入 {key_env}"
            await container.mount(
                Static(hint, classes="ob-hint"),
                Vertical(
                    Input(
                        placeholder="API Key（留空則沿用現有）" if existing_key else "API Key",
                        id="ob-key-input",
                        password=True,
                    ),
                    id="ob-key-section",
                ),
            )

        models = await resolve_dynamic_models(
            provider_id=self._selected_rag_pid or "",
            service="rag_llm",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "rag"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("  選擇模型：", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for m in models:
                await scroll.mount(
                    Button(
                        f"  {m}",
                        id=f"ob-ragmodel-{m.replace('/', '_').replace(':', '-').replace('.', '_')}",
                        classes="ob-prov-btn",
                        name=m,
                    ),
                )
        await container.mount(
            Horizontal(Button("<- 返回", id="btn-ob-back"), classes="ob-btn-row"),
        )

    async def _save_rag_config(self, model_name: str) -> None:
        from providers import RAG_LLM_PROVIDERS
        from settings import save_env_key

        pid = self._selected_rag_pid
        if not pid:
            return
        pinfo = RAG_LLM_PROVIDERS.get(pid, {})

        key_env = pinfo.get("key_env", "")
        if pinfo.get("auth_type") == "api_key" and key_env:
            # Always try to read the input field; use new value if provided
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val:
                    save_env_key(key_env, key_val)
            except Exception:
                pass

        prefix = pinfo.get("model_prefix", "")
        save_env_key("RAG_LLM_MODEL", prefix + model_name)

        if pinfo.get("default_base"):
            save_env_key("RAG_API_BASE", pinfo["default_base"])

        # Save RAG_API_KEY = same key as the provider key
        if key_env:
            from settings import load_env
            key_val = load_env().get(key_env, "")
            if key_val:
                save_env_key("RAG_API_KEY", key_val)

    # ── Step 4: Vision Model ──

    async def _render_vision_provider(self, container: Vertical) -> None:
        from providers import VISION_PROVIDERS
        await container.mount(
            Static("  Step 4/5: 圖片描述模型 (Vision)", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static("  圖片匯入時自動生成描述，需要支援 Vision 的模型", classes="ob-hint"),
            Static("", classes="ob-sep"),
        )
        for pid, pinfo in VISION_PROVIDERS.items():
            await container.mount(
                Button(
                    f"  {pinfo['label']}",
                    id=f"ob-vision-{pid}",
                    classes="ob-prov-btn",
                    name=pid,
                ),
            )

    async def _render_vision_setup(self, container: Vertical) -> None:
        from providers import VISION_PROVIDERS
        from config.service import get_config_service
        from model_registry import resolve_dynamic_models
        pinfo = VISION_PROVIDERS.get(self._selected_vision_pid, {})
        auth_type = pinfo.get("auth_type", "api_key")

        await container.mount(
            Static(f"  Step 4/5: {pinfo.get('label', '')} Vision Setup", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
        )

        # Check if key already exists from chat provider setup
        key_env = pinfo.get("key_env", "")
        if auth_type == "api_key" and key_env:
            from settings import load_env
            existing_key = load_env().get(key_env, "")
            hint = f"  已有 {key_env}（可直接選模型，或輸入新的 Key）" if existing_key else f"  請輸入 {key_env}"
            await container.mount(
                Static(hint, classes="ob-hint"),
                Vertical(
                    Input(
                        placeholder="API Key（留空則沿用現有）" if existing_key else "API Key",
                        id="ob-key-input",
                        password=True,
                    ),
                    id="ob-key-section",
                ),
            )

        # Model selection
        models = await resolve_dynamic_models(
            provider_id=self._selected_vision_pid or "",
            service="vision",
            fallback=list(pinfo.get("models", [])),
            env=get_config_service().snapshot(include_process=True),
        )
        self._model_candidates = list(models)
        self._model_button_kind = "vision"
        models = self._filter_models(self._model_candidates)
        if models:
            await container.mount(
                Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
                Static("  選擇 Vision 模型", classes="ob-hint"),
                Input(
                    value=self._model_search_query,
                    placeholder="搜尋模型...",
                    id="ob-model-search",
                ),
            )
            scroll = VerticalScroll(id="ob-model-scroll")
            await container.mount(scroll)
            for idx, model in enumerate(models):
                await scroll.mount(
                    Button(
                        f"  {model}",
                        id=f"ob-vmodel-{idx}",
                        classes="ob-prov-btn",
                        name=model,
                    ),
                )
        await container.mount(
            Horizontal(
                Button("<- 返回", id="btn-ob-back"),
                classes="ob-btn-row",
            ),
        )

    async def _render_done(self, container: Vertical) -> None:
        from settings import load_env
        env = load_env()
        model = env.get("LLM_MODEL", "(未設定)")
        embed = env.get("EMBED_MODEL", "")
        if not embed or "/" not in embed:
            embed = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
        rag_llm = env.get("RAG_LLM_MODEL", "")
        rag_display = rag_llm if rag_llm else "跟聊天模型相同"
        vision = env.get("VISION_MODEL", "")
        vision_display = vision if vision else "(未設定)"

        await container.mount(
            Static("  Step 5/5: 設定完成!", classes="ob-title"),
            Static("[dim #2a2a2a]" + "-" * 56 + "[/]", classes="ob-sep"),
            Static(f"  Chat: {model}", classes="ob-hint"),
            Static(f"  Embedding: {embed}", classes="ob-hint"),
            Static(f"  知識庫 LLM: {rag_display}", classes="ob-hint"),
            Static(f"  Vision: {vision_display}", classes="ob-hint"),
            Static("", classes="ob-sep"),
            Static("  可隨時用 Ctrl+S 開啟設定修改", classes="ob-hint"),
            Horizontal(
                Button("開始使用 ->", id="btn-ob-finish"),
                classes="ob-btn-row",
            ),
        )

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        btn_name = event.button.name or ""

        # Step 1: chat provider selection
        if btn_id.startswith("ob-chat-"):
            self._selected_chat_pid = btn_name
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "chat_setup"
            await self._render_step()
            return

        # Step 1 setup: model selection
        if btn_id.startswith("ob-model-"):
            await self._save_chat_config(btn_name)
            self._step = "embed"
            await self._render_step()
            return

        # Step 1 setup: next (for providers without model list)
        if btn_id == "btn-ob-next":
            await self._save_chat_config(None)
            self._step = "embed"
            await self._render_step()
            return

        # Step 3: RAG LLM provider selection
        if btn_id.startswith("ob-rag-"):
            pid = btn_name
            if pid == "same-as-chat":
                from settings import save_env_key
                save_env_key("RAG_LLM_MODEL", "")
                self._step = "vision_provider"
                await self._render_step()
                return
            self._selected_rag_pid = pid
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "rag_setup"
            await self._render_step()
            return

        # Step 4: Vision provider selection
        if btn_id.startswith("ob-vision-"):
            pid = btn_name
            if pid == "skip-vision":
                self._step = "done"
                await self._render_step()
                return
            self._selected_vision_pid = pid
            self._model_search_query = ""
            self._model_candidates = []
            self._model_button_kind = None
            self._step = "vision_setup"
            await self._render_step()
            return

        # Step 4: Vision model selection
        if btn_id.startswith("ob-vmodel-"):
            await self._save_vision_config(btn_name)
            self._step = "done"
            await self._render_step()
            return

        # Step 3: RAG model selection
        if btn_id.startswith("ob-ragmodel-"):
            await self._save_rag_config(btn_name)
            self._step = "vision_provider"
            await self._render_step()
            return

        # Back button
        if btn_id == "btn-ob-back":
            if self._step == "chat_setup":
                self._model_search_query = ""
                self._step = "chat_provider"
            elif self._step == "embed":
                self._step = "chat_provider"
            elif self._step == "embed_setup":
                self._step = "embed"
            elif self._step == "rag_llm":
                self._step = "embed"
            elif self._step == "rag_setup":
                self._model_search_query = ""
                self._step = "rag_llm"
            elif self._step == "vision_provider":
                self._step = "rag_llm"
            elif self._step == "vision_setup":
                self._model_search_query = ""
                self._step = "vision_provider"
            await self._render_step()
            return

        # Step 2: embed provider selection
        if btn_id.startswith("ob-embed-provider-"):
            embed_pid = btn_name
            if embed_pid == "openrouter":
                self._selected_embed_pid = "openrouter"
                self._step = "embed_setup"
                await self._render_step()
                return

        # Step 2: embedding API key save
        if btn_id == "btn-ob-embed-save":
            await self._save_embedding_config()
            self._step = "rag_llm"
            await self._render_step()
            return

        # Step 2: embed — keep backward-compatible button ids
        if btn_id == "ob-embed-download":
            await self._save_embedding_config()
            self._step = "embed_download"
            await self._render_step()
            return

        if btn_id == "ob-embed-skip":
            await self._save_embedding_config()
            self._step = "rag_llm"
            await self._render_step()
            return

        # Finish
        if btn_id == "btn-ob-finish":
            self.dismiss("done")
            return

    async def _save_chat_config(self, model_name: str | None) -> None:
        from providers import CHAT_PROVIDERS
        from settings import save_env_key

        pid = self._selected_chat_pid
        if not pid:
            return

        pinfo = CHAT_PROVIDERS.get(pid, {})

        # Save API key if provided
        if pinfo.get("auth_type") not in ("none", "oauth"):
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val and pinfo.get("key_env"):
                    save_env_key(pinfo["key_env"], key_val)
            except Exception:
                pass

        # Save base URL if provider has default
        if pinfo.get("default_base"):
            base_env = pinfo.get("base_env", "OPENAI_API_BASE")
            if base_env:
                save_env_key(base_env, pinfo["default_base"])

        # Save model
        prefix = pinfo.get("model_prefix", "")
        if model_name:
            full_model = prefix + model_name
            save_env_key("LLM_MODEL", full_model)
        elif prefix:
            # No model list (e.g. codex-cli/) — use prefix + default
            default = "codex-mini-latest" if "codex-cli" in pid else "default"
            save_env_key("LLM_MODEL", prefix + default)

    async def _save_vision_config(self, model_name: str) -> None:
        from providers import VISION_PROVIDERS
        from settings import save_env_key

        pid = self._selected_vision_pid
        if not pid:
            return
        pinfo = VISION_PROVIDERS.get(pid, {})

        # Save API key if provided (new value overrides existing)
        key_env = pinfo.get("key_env", "")
        if pinfo.get("auth_type") == "api_key" and key_env:
            try:
                key_input = self.query_one("#ob-key-input", Input)
                key_val = key_input.value.strip()
                if key_val:
                    save_env_key(key_env, key_val)
            except Exception:
                pass

        # Save VISION_MODEL with prefix
        prefix = pinfo.get("model_prefix", "")
        save_env_key("VISION_MODEL", prefix + model_name)

        # Save VISION_API_BASE if provider has a default
        if pinfo.get("default_base"):
            save_env_key("VISION_API_BASE", pinfo["default_base"])

    async def _save_embed_config(self, embed_pid: str) -> None:
        # v0.7: 向後相容 shim
        from settings import save_env_key
        save_env_key("EMBED_PROVIDER", "openrouter")
        save_env_key("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
        save_env_key("EMBED_DIM", "1024")
        save_env_key("EMBED_API_BASE", "https://openrouter.ai/api/v1")

    async def _save_embedding_config(self) -> None:
        """儲存 OpenRouter embedding 設定。"""
        from settings import save_env_key
        try:
            key_input = self.query_one("#ob-key-input", Input)
            key_val = key_input.value.strip()
            if key_val:
                save_env_key("OPENROUTER_API_KEY", key_val)
                save_env_key("EMBED_API_KEY", key_val)
        except Exception:
            pass
        save_env_key("EMBED_PROVIDER", "openrouter")
        save_env_key("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")
        save_env_key("EMBED_DIM", "1024")
        save_env_key("EMBED_API_BASE", "https://openrouter.ai/api/v1")
        save_env_key("EMBED_DIM", "1024")