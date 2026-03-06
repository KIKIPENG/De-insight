"""De-insight TUI — Elia 風格的對話介面"""

import json
import sys
from pathlib import Path

# Allow importing from backend/
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import httpx
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Input, Markdown, Static

from codex_client import codex_stream, is_codex_available
from settings import SettingsScreen, load_env

SPINNER_FRAMES = ["|", "/", "—", "\\"]


# ── Widgets ──────────────────────────────────────────────────────────


class ModeIndicator(Static):
    """模式指示器。"""

    pass


class WelcomeBlock(Vertical):
    """初始歡迎畫面。"""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold #fafafa]◈ De-insight[/]  [dim]批判性對話者[/]",
        )
        yield Static(
            "[dim #6e7681]────────────────────────────[/]",
        )
        yield Static(
            "[#8b949e]△ 我不是助手。我是挑戰者。\n"
            "△ 質疑你的視覺決策背後的權力結構。\n"
            "△ 基於 Foucault 規訓理論框架。[/]",
        )
        yield Static(
            "[dim #484f58]⌘ ctrl+s 設定 ∙ ctrl+e 模式 ∙ ctrl+n 新對話[/]",
        )


class ThinkingIndicator(Static):
    """斜線旋轉思考指示器。"""

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self._frame = 0
        self._timer: Timer | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._spin)

    def _spin(self) -> None:
        ch = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
        self.update(Text(f" {ch} ", style="#fafafa"))
        self._frame += 1

    def stop(self) -> None:
        if self._timer:
            self._timer.stop()


class Chatbox(Vertical):
    """Elia 風格的訊息框：圓角邊框 + 標題。"""

    # Breathing animation: cycle border between dim and bright
    _BREATH_COLORS = [
        "#2a2a2a", "#3a3a3a", "#4a4a4a", "#5a5a5a",
        "#6a6a6a", "#7a7a7a", "#8a8a8a", "#9a9a9a",
        "#aaaaaa", "#bbbbbb", "#cccccc", "#dddddd",
        "#eeeeee", "#fafafa",
        "#eeeeee", "#dddddd", "#cccccc", "#bbbbbb",
        "#aaaaaa", "#9a9a9a", "#8a8a8a", "#7a7a7a",
        "#6a6a6a", "#5a5a5a", "#4a4a4a", "#3a3a3a",
    ]

    def __init__(self, role: str, content: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self._body = Markdown(content, classes="chatbox-body")
        self.add_class(f"chatbox-{role}")
        self._breath_timer: Timer | None = None
        self._breath_frame = 0

    def on_mount(self) -> None:
        if self.role == "user":
            self.border_title = "◇ You"
        else:
            self.border_title = "◆ De-insight"

    def compose(self) -> ComposeResult:
        yield self._body

    def stream_update(self, content: str) -> None:
        self._body.update(content)

    def _breathe(self) -> None:
        color = self._BREATH_COLORS[self._breath_frame % len(self._BREATH_COLORS)]
        self.styles.border = ("round", color)
        self.styles.border_title_color = color
        self._breath_frame += 1

    def set_responding(self, responding: bool) -> None:
        if responding:
            self.border_title = "◆ De-insight"
            self.add_class("responding")
            self._breath_frame = 0
            if self.is_mounted:
                self._breath_timer = self.set_interval(0.08, self._breathe)
        else:
            if self._breath_timer:
                self._breath_timer.stop()
                self._breath_timer = None
            self.border_title = "◆ De-insight"
            self.remove_class("responding")
            # Reset — use rgba hex (no "40%" syntax in Python API)
            self.styles.border = ("round", "#666666")
            self.styles.border_title_color = "#fafafa"


class StatusBar(Static):
    """底部狀態列。"""

    pass


# ── App ──────────────────────────────────────────────────────────────


class DeInsightApp(App):
    TITLE = "De-insight"
    CSS = """
    Screen {
        background: #0a0a0a;
        color: #fafafa;
    }

    /* ── mode indicator ── */
    ModeIndicator {
        dock: top;
        height: 1;
        padding: 0 2;
        background: #0a0a0a;
        color: #6e7681;
        border-bottom: solid #2a2a2a;
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
        border: round #3a3a3a;
        border-title-color: #8b949e;
    }

    /* ── chatbox ── */
    Chatbox {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #2a2a2a;
        border-title-color: #6e7681;
    }

    .chatbox-user {
        border: round #3a3a3a;
        border-title-color: #8b949e;
    }

    .chatbox-assistant {
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    .chatbox-assistant.responding {
        background: #fafafa 3%;
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

    /* ── markdown overrides ── */
    Markdown {
        margin: 0;
        padding: 0;
        background: transparent;
    }

    MarkdownH1, MarkdownH2, MarkdownH3 {
        margin: 0;
        padding: 0;
        background: transparent;
        color: #fafafa;
    }

    MarkdownFence {
        margin: 1 0;
        padding: 1 2;
        background: #111111;
        color: #e6edf3;
    }

    MarkdownBlockQuote {
        margin: 0;
        padding: 0 0 0 2;
        border-left: tall #fafafa;
        background: transparent;
        color: #8b949e;
    }

    MarkdownBulletList, MarkdownOrderedList {
        margin: 0;
        padding: 0 0 0 2;
    }

    /* ── thinking indicator ── */
    ThinkingIndicator {
        height: 1;
        margin: 1 1 0 1;
        padding: 0 2;
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    /* ── input area ── */
    #input-box {
        dock: bottom;
        height: auto;
        max-height: 8;
        padding: 0 2 0 2;
        margin: 0;
        background: #0a0a0a;
    }

    #input-frame {
        height: auto;
        margin: 0 1;
        padding: 0 1;
        border: round #3a3a3a;
        border-title-color: #6e7681;
        background: #111111;
    }

    #input-frame:focus-within {
        border: round #fafafa 40%;
        border-title-color: #fafafa;
    }

    #chat-input {
        background: transparent;
        color: #fafafa;
        border: none;
        padding: 0;
        margin: 0;
    }

    /* ── status bar ── */
    StatusBar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: #111111;
        color: #6e7681;
    }

    /* ── footer ── */
    Footer {
        background: #111111;
        color: #6e7681;
    }

    Footer > .footer--highlight {
        background: #fafafa;
        color: #0a0a0a;
    }

    Footer > .footer--key {
        background: #1a1a1a;
        color: #fafafa;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "open_settings", "⚙ 設定", priority=True),
        Binding("ctrl+e", "toggle_mode", "感性/理性", priority=True),
        Binding("ctrl+n", "new_chat", "新對話", priority=True),
        Binding("ctrl+c", "quit", "退出"),
    ]

    mode: reactive[str] = reactive("emotional")
    is_loading: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[dict] = []
        self.api_base = "http://localhost:8000"

    def compose(self) -> ComposeResult:
        yield ModeIndicator(id="mode-indicator")
        yield VerticalScroll(
            Vertical(id="messages"),
            id="chat-scroll",
        )
        yield StatusBar(id="status-bar")
        input_frame = Vertical(
            Input(placeholder="輸入你的想法…", id="chat-input"),
            id="input-frame",
        )
        input_frame.border_title = "⌨ Message"
        yield Vertical(input_frame, id="input-box")
        yield Footer()

    async def on_mount(self) -> None:
        self._update_mode_indicator()
        self._update_status()
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.1"
        await container.mount(welcome)
        self.query_one("#chat-input", Input).focus()

    # ── status ──

    def watch_mode(self) -> None:
        if self.is_mounted:
            self._update_mode_indicator()
            self._update_status()

    def watch_is_loading(self) -> None:
        if self.is_mounted:
            self._update_status()

    def _update_mode_indicator(self) -> None:
        e_sym = "●" if self.mode == "emotional" else "○"
        r_sym = "●" if self.mode == "rational" else "○"
        emotional = (
            f"[bold #fafafa]{e_sym} 感性[/]"
            if self.mode == "emotional"
            else f"[#484f58]{e_sym} 感性[/]"
        )
        rational = (
            f"[bold #fafafa]{r_sym} 理性[/]"
            if self.mode == "rational"
            else f"[#484f58]{r_sym} 理性[/]"
        )
        try:
            self.query_one("#mode-indicator", ModeIndicator).update(
                f"  {emotional}  [dim #3a3a3a]│[/]  {rational}"
                f"    [dim #3a3a3a]ctrl+e[/]"
            )
        except NoMatches:
            pass

    def _update_status(self) -> None:
        mode_label = "感性" if self.mode == "emotional" else "理性"
        loading = "  [italic #fafafa]⟳ thinking…[/]" if self.is_loading else ""
        msg_count = len(self.messages)
        env = load_env()
        model = env.get("LLM_MODEL", "?")
        if model.startswith("codex-cli/"):
            backend_label = "◈ Codex CLI"
        else:
            backend_label = f"◇ {model}"
        try:
            self.query_one("#status-bar", StatusBar).update(
                f"[#484f58]{backend_label}[/]"
                f"  [dim #3a3a3a]│[/]  "
                f"[#484f58]◇ {mode_label}[/]"
                f"  [dim #3a3a3a]│[/]  "
                f"[#484f58]{msg_count} msgs[/]"
                f"{loading}"
            )
        except NoMatches:
            pass

    # ── actions ──

    def action_toggle_mode(self) -> None:
        self.mode = "rational" if self.mode == "emotional" else "emotional"

    def action_open_settings(self) -> None:
        def on_dismiss(result: str | None) -> None:
            if result:
                self._update_status()
                self._reload_backend_env()

        self.push_screen(SettingsScreen(), callback=on_dismiss)

    @work(thread=True)
    async def _reload_backend_env(self) -> None:
        """通知後端重新載入 .env。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{self.api_base}/api/reload-env")
                if resp.status_code == 200:
                    data = resp.json()
                    model = data.get("model", "?")
                    self.notify(f"⚙ 設定已儲存 ∙ 模型: {model}")
                else:
                    self.notify("⚙ 設定已儲存（後端重載失敗，請手動重啟）")
        except Exception:
            self.notify("⚙ 設定已儲存到 .env（後端未運行）")

    def action_new_chat(self) -> None:
        self.messages.clear()
        container = self.query_one("#messages", Vertical)
        container.remove_children()
        self.call_after_refresh(self._mount_welcome)
        self.query_one("#chat-input", Input).focus()

    async def _mount_welcome(self) -> None:
        container = self.query_one("#messages", Vertical)
        welcome = WelcomeBlock()
        welcome.border_title = "◈ De-insight v0.1"
        await container.mount(welcome)

    # ── chat ──

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text or self.is_loading:
            return

        event.input.value = ""
        self.messages.append({"role": "user", "content": text})

        container = self.query_one("#messages", Vertical)

        for w in container.query("WelcomeBlock"):
            await w.remove()

        await container.mount(Chatbox("user", text))
        self._scroll_to_bottom()
        self._stream_response()

    def _is_codex_cli_mode(self) -> bool:
        """檢查目前是否使用 Codex CLI 模式。"""
        env = load_env()
        return env.get("LLM_MODEL", "").startswith("codex-cli/")

    @work(exclusive=True)
    async def _stream_response(self) -> None:
        self.is_loading = True
        container = self.query_one("#messages", Vertical)

        # 立即建立 AI 回覆框，呼吸燈從問問題就開始
        bubble = Chatbox("assistant")
        await container.mount(bubble)
        bubble.set_responding(True)
        self._scroll_to_bottom()

        full_content = ""

        try:
            if self._is_codex_cli_mode():
                # ── Codex CLI 路徑 ──
                if not is_codex_available():
                    raise RuntimeError("codex CLI 未安裝。執行: npm i -g @openai/codex")

                from prompts.foucault import get_system_prompt
                sys_prompt = get_system_prompt(self.mode)
                user_msg = self.messages[-1]["content"]

                # Include conversation context
                context = ""
                for m in self.messages[:-1]:
                    role = "You" if m["role"] == "user" else "De-insight"
                    context += f"{role}: {m['content']}\n\n"

                full_prompt = f"{context}You: {user_msg}" if context else user_msg

                env = load_env()
                codex_model = env.get("LLM_MODEL", "").removeprefix("codex-cli/")

                async for chunk in codex_stream(full_prompt, sys_prompt, model=codex_model):
                    full_content += chunk
                    bubble.stream_update(full_content)
                    self._scroll_to_bottom()
            else:
                # ── FastAPI 後端路徑 ──
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST",
                        f"{self.api_base}/api/chat",
                        json={"messages": self.messages, "mode": self.mode},
                    ) as response:
                        async for line in response.aiter_lines():
                            if line.startswith("0:"):
                                try:
                                    chunk = json.loads(line[2:])
                                    full_content += chunk
                                    bubble.stream_update(full_content)
                                    self._scroll_to_bottom()
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            elif line.startswith("3:"):
                                try:
                                    err = json.loads(line[2:])
                                    raise RuntimeError(err.get("error", "未知錯誤"))
                                except json.JSONDecodeError:
                                    raise RuntimeError(line[2:])
                            elif line.startswith("d:"):
                                break

            bubble.set_responding(False)
            self.messages.append({"role": "assistant", "content": full_content})

        except httpx.ConnectError:
            bubble.set_responding(False)
            bubble.stream_update(
                "**連線錯誤** — 後端未啟動\n\n"
                "```\ncd backend && .venv/bin/python3 -m uvicorn main:app --reload\n```"
            )
        except Exception as e:
            bubble.set_responding(False)
            bubble.stream_update(f"**錯誤** — {escape(str(e))}")
        finally:
            self.is_loading = False
            self._update_status()
            self.query_one("#chat-input", Input).focus()

    def _scroll_to_bottom(self) -> None:
        self.query_one("#chat-scroll", VerticalScroll).scroll_end(animate=False)


if __name__ == "__main__":
    DeInsightApp().run()
