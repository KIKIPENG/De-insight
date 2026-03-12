"""Mixin Protocol — 定義所有 Mixin 對 App 的依賴介面。

讓 IDE 與 type checker 能正確推斷跨 Mixin 呼叫。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from textual.reactive import reactive
    from widgets import AppState
    from conversation.store import ConversationStore
    from projects.manager import ProjectManager


@runtime_checkable
class DeInsightProtocol(Protocol):
    """App 必須提供的介面，供各 Mixin 使用。

    此 Protocol 定義了所有 Mixin（ChatMixin、MemoryMixin、RAGMixin、ProjectMixin、UIMixin）
    對 DeInsightApp 實例的期望介面。包括：
    - Reactive 狀態屬性
    - 資料存儲和管理器
    - 跨 Mixin 共享的公共方法

    任何新增的 Mixin 如果使用這裡未列出的屬性或方法，應該先更新此 Protocol。
    """

    # ── Reactive Properties ──
    mode: str
    """當前對話模式：'emotional'（感性）或 'rational'（理性）"""

    rag_mode: str
    """RAG 檢索模式：'fast'（快速）或 'deep'（深度）"""

    is_loading: bool
    """是否正在載入（LLM 串流進行中）"""

    # ── State & Collections ──
    messages: list[dict]
    """當前對話的訊息列表"""

    state: AppState
    """全域應用狀態：當前專案、待確認記憶、互動深度等"""

    # ── Stores & Managers ──
    _conv_store: ConversationStore
    """對話歷史持久化存儲"""

    _project_manager: ProjectManager
    """專案 CRUD 管理"""

    api_base: str
    """FastAPI 後端 base URL（通常為 'http://localhost:8000'）"""

    # ── LLM Call Tracking ──
    _llm_call_count: int
    """LLM 呼叫次數計數（用於狀態列顯示）"""

    # ── Cross-Mixin Methods ──
    def _update_menu_bar(self) -> None:
        """更新菜單列（UIMixin）

        重新渲染菜單列的動態部分，包括當前模式、RAG 狀態、記憶計數等。
        """
        ...

    def _update_status(self) -> None:
        """更新狀態列（UIMixin）

        重新渲染狀態列，包括模式、RAG 模式、LLM 呼叫數、知識圖譜狀態等。
        """
        ...

    def _scroll_to_bottom(self) -> None:
        """滾動對話區域到底部（UIMixin）

        保持對話視圖始終顯示最新訊息。
        """
        ...

    async def _refresh_memory_panel(self) -> None:
        """重新整理記憶側邊欄（MemoryMixin）

        從資料庫載入並重新渲染記憶面板的內容。
        """
        ...

    async def _quick_llm_call(
        self, prompt: str, max_tokens: int = 500, max_retries: int = 3
    ) -> str:
        """快速 LLM 呼叫（ChatMixin）

        用於內部邏輯計算，不走對話流程。直接調用後端 /api/chat 並等待完整響應。
        會自動更新 _llm_call_count。

        Args:
            prompt: 提示詞
            max_tokens: 最大回覆 token 數
            max_retries: 失敗重試次數

        Returns:
            LLM 回覆文本
        """
        ...

    def _start_discussion_from_memory(self, memory_content: str) -> None:
        """從記憶啟動新討論（UIMixin）

        將記憶內容注入對話輸入框，並提示使用者提出相關問題。
        """
        ...

    def action_close_modals(self) -> None:
        """關閉所有開放的 Modal（UIMixin）

        Textual action，由 binding 或 action_name 觸發。
        """
        ...

    def fill_input(self, text: str) -> None:
        """填充對話輸入框（ChatMixin）

        將文本寫入 ChatInput 並聚焦，不自動提交。
        """
        ...

    async def _show_system_message(self, content: str) -> None:
        """顯示系統訊息（UIMixin）

        在對話區域顯示系統提示訊息，通常用於狀態更新或錯誤提示。
        """
        ...

    def notify(
        self,
        message: str,
        *,
        severity: str = "information",
        timeout: float = 5.0,
    ) -> None:
        """顯示通知（來自 App 基類）

        在螢幕下方顯示短暫通知。

        Args:
            message: 通知文本
            severity: 嚴重級別（'information'、'warning'、'error'）
            timeout: 顯示秒數
        """
        ...
