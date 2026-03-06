# De-insight v0.2 — 自動 Debug 指示

給 Claude Code 的逐步除錯任務。每個步驟執行完才進行下一步。

---

## 執行前提示

每個步驟開始前：
1. 執行指定的測試指令
2. 記錄實際錯誤訊息到 `progress.md`
3. 修復
4. 再次執行測試確認通過
5. 才進入下一步

---

## 步驟一：確認啟動不崩潰

### 測試
```bash
cd project-root
python tui.py 2>&1 | head -30
```

### 已知問題一：`panels` 模組缺失或有錯

`tui.py` 的 import：
```python
from panels import (
    ImportModal, InsightConfirmModal, MemoryManageModal,
    MemoryPanel, ResearchPanel, SearchModal,
)
```

**如果 `panels.py` 不存在**，建立它，包含以下六個 class：

**`ResearchPanel`（Static 子類）**
```python
class ResearchPanel(Static):
    def compose(self):
        yield Static(
            "[dim #484f58]ctrl+f 匯入 PDF / 網頁\nctrl+k 搜尋[/]",
            id="research-content",
        )
```

**`MemoryPanel`（Static 子類）**
```python
class MemoryPanel(Static):
    def compose(self):
        yield Static(
            "[dim #484f58]對話後自動記錄洞見[/]",
            id="memory-content",
        )
```

**`ImportModal`（ModalScreen 子類）**
- 輸入框接受 PDF 路徑或 URL
- 按下 Enter 或「匯入」按鈕後 `self.dismiss(path)` 
- Escape 取消 `self.dismiss(None)`

**`SearchModal`（ModalScreen 子類）**
- 輸入框接受搜尋關鍵字
- 按下 Enter 後 `self.dismiss(query)`
- Escape 取消 `self.dismiss(None)`

**`MemoryManageModal`（ModalScreen 子類）**
- 列出所有記憶（呼叫 `asyncio.get_event_loop().run_until_complete(get_memories())` 或用 `on_mount` + `@work` 載入）
- 每條記憶旁有「刪除」按鈕，呼叫 `delete_memory(id)`
- 有篩選類型的按鈕（全部 / 洞見 / 問題 / 反應）
- Escape 關閉

**`InsightConfirmModal`（ModalScreen 子類）**
- 接收 `draft: str` 和 `type: str` 兩個參數
- 顯示 LLM 生成的洞見草稿
- 可編輯內容
- 有「儲存」按鈕 → `self.dismiss({"type": type, "content": edited_content})`
- 有「取消」按鈕 → `self.dismiss(None)`

---

### 已知問題二：`knowledge_graph.py` 的 settings import 路徑錯誤

```python
# 現在（錯誤）
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
from settings import load_env

# 應該改為（settings.py 在專案根目錄）
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import load_env
```

---

### 已知問題三：LightRAG import 路徑可能不存在

```python
# 現在
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
```

用以下指令確認 lightrag-hku 實際的 import 路徑：
```bash
python -c "import lightrag; print(lightrag.__file__)"
python -c "from lightrag.llm.openai import openai_complete_if_cache; print('OK')"
```

如果失敗，改用 litellm 統一介面：
```python
import litellm

async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for h in (history_messages or []):
        msgs.append(h)
    msgs.append({"role": "user", "content": prompt})
    resp = await litellm.acompletion(
        model=llm_model,
        messages=msgs,
        **kwargs,
    )
    return resp.choices[0].message.content or ""

async def embed_func(texts):
    resp = await litellm.aembedding(
        model=embed_model,
        input=texts,
    )
    return [item["embedding"] for item in resp.data]
```

---

### 驗收
```bash
python tui.py
```
TUI 啟動，左右分割佈局可見，沒有 traceback。

---

## 步驟二：對話功能正常

### 測試
1. 啟動 backend：`cd backend && uvicorn main:app --reload`
2. 啟動 TUI：`python tui.py`
3. 輸入一句話，確認 AI 有回應

### 已知問題一：`_stream_response` 裡的 `send_messages`

```python
send_messages = await self._inject_rag_context(self.messages)
```

`_inject_rag_context` 若 LightRAG 初始化失敗會拋出 exception 導致整個 `_stream_response` 崩潰。確認 except 有正確 fallback：

```python
async def _inject_rag_context(self, messages):
    try:
        from rag.knowledge_graph import query_knowledge, has_knowledge
        if not has_knowledge():
            return messages
        # ... 其餘邏輯
    except Exception:
        return messages  # 永遠 fallback，不讓 RAG 錯誤中斷對話
```

### 已知問題二：Codex CLI 的 `model` 參數

`tui.py` 呼叫：
```python
async for chunk in codex_stream(full_prompt, sys_prompt, model=codex_model)
```

但 `codex_client.py` 的 `codex_stream` 函數簽名：
```python
async def codex_stream(prompt: str, system_prompt: str = "") -> AsyncGenerator[str, None]:
```

沒有 `model` 參數。加入：
```python
async def codex_stream(
    prompt: str,
    system_prompt: str = "",
    model: str = "codex-mini-latest",
) -> AsyncGenerator[str, None]:
    cmd = ["codex", "exec", "--json", "--model", model]
    ...
```

### 驗收
輸入「什麼是包豪斯？」，AI 用繁體中文以 Foucault 框架回應。

---

## 步驟三：知識庫匯入功能

### 測試
1. 按 `ctrl+f`，確認 modal 出現
2. 輸入一個存在的 PDF 路徑
3. 確認有「匯入中…」通知
4. 完成後研究面板更新

### 已知問題一：`MemoryManageModal` 的 async 載入

`MemoryManageModal.on_mount` 不能直接 `await`，必須用 `@work`：

```python
class MemoryManageModal(ModalScreen):
    def on_mount(self):
        self._load_memories()

    @work()
    async def _load_memories(self):
        from memory.store import get_memories
        mems = await get_memories()
        # 更新 UI
```

### 已知問題二：`insert_pdf` 需要 pypdf

```bash
pip install pypdf
```

確認 `requirements.txt` 有加：
```
pypdf>=4.0.0
```

### 已知問題三：Ollama embedding 模型

如果用 Ollama，`nomic-embed-text` 要先 pull：
```bash
ollama pull nomic-embed-text
```

如果沒有這個模型，`insert_pdf` 會靜默失敗。加入更明確的錯誤訊息：
```python
# knowledge_graph.py 的 embed_func
async def embed_func(texts):
    try:
        return await openai_embed(...)
    except Exception as e:
        raise RuntimeError(f"Embedding 失敗，請確認模型已安裝: {e}") from e
```

### 驗收
匯入一個小 PDF（< 10 頁），研究面板出現「匯入完成，知識庫已更新」，`data/lightrag/` 目錄有新檔案。

---

## 步驟四：RAG 查詢注入對話

### 測試
1. 匯入 PDF 後
2. 問一個跟文件相關的問題
3. 研究面板顯示查詢結果
4. AI 回應有引用文件內容的痕跡

### 已知問題一：`has_knowledge()` 的判斷邏輯

```python
def has_knowledge() -> bool:
    graph_file = WORKING_DIR / "graph_chunk_entity_relation.graphml"
    return graph_file.exists() and graph_file.stat().st_size > 100
```

這個檔名在 lightrag-hku 的不同版本可能不一樣。更健壯的寫法：

```python
def has_knowledge() -> bool:
    # 檢查任何非空的 graphml 或 json 圖檔
    for pattern in ["*.graphml", "kv_store_*.json"]:
        for f in WORKING_DIR.glob(pattern):
            if f.stat().st_size > 200:
                return True
    return False
```

### 已知問題二：RAG 結果截斷太短

```python
# 現在
rag_context = {
    "role": "system",
    "content": f"以下是從知識庫中找到的相關資訊：\n\n{result[:1000]}",
}
```

1000 字元可能截斷了最重要的內容。調整為：
```python
content=f"以下是從知識庫中找到的相關資訊，可選擇性參考：\n\n{result[:2000]}"
```

### 已知問題三：研究面板更新時機

`_update_research_panel` 是 `async def`，但在 `_stream_response` 裡用 `await` 呼叫沒問題。確認 `_inject_rag_context` 有正確 `await` 它：

```python
if result and len(result.strip()) > 10:
    await self._update_research_panel(result)  # ← 必須 await
```

### 驗收
問一個文件裡有的問題，研究面板有內容，AI 回答比沒有文件時更具體。

---

## 步驟五：記憶自動抽取

### 測試
1. 說幾句有實質內容的話，例如：
   - 「我覺得極簡主義其實是一種焦慮的掩蓋」
   - 「包豪斯的功能主義讓我感到窒息」
2. 對話結束後 2-3 秒，觀察記憶面板有沒有出現新條目

### 已知問題一：`extract_memories` 對閒聊過於敏感

如果使用者說「你好」或「謝謝」，應該回傳空陣列。確認 prompt 有明確排除：
```python
EXTRACT_PROMPT = """
...
不要記錄：
- 問候語、感謝語、閒聊
- 純粹是回應 AI 的話（「嗯嗯」「好的」「我懂了」）
- 問 AI 的問題（這是 AI 的任務，不是洞見）
...
"""
```

### 已知問題二：小模型的 JSON 輸出不穩定

小模型（llama3.2、qwen2.5）有時回傳被 markdown 包住的 JSON：
```
```json
[{"type": "insight", "content": "..."}]
```
```

`extract_memories` 的 JSON 解析需要先清理：
```python
def _clean_json(text: str) -> str:
    text = text.strip()
    # 移除 markdown code block
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return text.strip()

# 在 extract_memories 裡
try:
    items = json.loads(_clean_json(response.strip()))
```

### 已知問題三：`_background_memory_extract` 的 llm_call 模型選擇

```python
if model.startswith("codex-cli/"):
    model = "ollama/llama3.2"
```

如果使用者沒有安裝 Ollama，這裡會失敗。改為：
```python
if model.startswith("codex-cli/"):
    # codex-cli 無法用於記憶抽取，嘗試用 Ollama，失敗就跳過
    try:
        model = "ollama/llama3.2"
        # test if ollama is running
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        r.raise_for_status()
    except Exception:
        return  # 跳過記憶抽取
```

### 驗收
說「我覺得 Judd 的極簡主義讓我感到安靜但不空洞」後，記憶面板出現 💭 的條目。

---

## 步驟六：記憶管理 modal

### 測試
1. 按 `ctrl+m` 開啟記憶管理
2. 確認有記憶列表
3. 刪除一條記憶
4. 關閉後記憶面板更新

### 已知問題一：刪除後 UI 沒有立即更新

`MemoryManageModal` 刪除後要重新載入列表：
```python
async def _delete_memory(self, memory_id: int) -> None:
    from memory.store import delete_memory
    await delete_memory(memory_id)
    await self._load_memories()  # 重新載入
```

### 已知問題二：modal 裡的 async 呼叫需要用 `@work`

所有 `async def` 在 Textual widget 裡不能直接呼叫，必須用 `@work` 裝飾器或 `self.call_later`。

### 驗收
刪除記憶後列表立即更新，關閉 modal 後記憶面板也更新。

---

## 步驟七：思維演變偵測

### 測試
1. 先說：「我覺得極簡主義很美」
2. 等記憶被儲存
3. 幾輪對話後說：「我現在覺得極簡主義其實是一種暴力」
4. 觀察是否出現演變通知

### 已知問題一：`check_for_evolution` 的 search 關鍵字太短

`search_memories(new_insight, limit=5)` 用整個新洞見做 LIKE 查詢，通常不會有結果。

改為取前 20 字：
```python
search_query = new_insight[:20]
related = await search_memories(search_query, limit=5)
```

或改為取關鍵詞（拆分斷詞後取最有意義的詞）。

### 已知問題二：小模型不一定能輸出有效 JSON

在 `check_for_evolution` 加同樣的 `_clean_json` 處理：
```python
try:
    result = json.loads(_clean_json(response.strip()))
```

### 已知問題三：演變通知訊息過長

```python
self.notify(
    f"思維{evolution.get('type', '?')}: {evolution.get('summary', '')[:50]}"
)
```

Textual 的 `notify` 在長字串時會截斷，改為：
```python
etype = "演變" if evolution.get("type") == "evolution" else "矛盾"
self.notify(f"⟳ 偵測到思維{etype}", title="思維追蹤", timeout=5)
```

### 驗收
偵測到矛盾時，右上角出現通知，記憶面板顯示特殊標記。

---

## 步驟八：`[save insight]` 按鈕

### 測試
1. 對話幾輪
2. 點擊 AI 回覆下方的 `[save insight]`
3. 確認 `InsightConfirmModal` 出現，顯示 LLM 生成的洞見草稿
4. 確認可以編輯並儲存

### 已知問題一：`action_save_insight_from_chat` 的 chatbox 索引邏輯有 bug

```python
# 現在（有問題）
for i, box in enumerate(boxes):
    if box is chatbox:
        ai_msg = self.messages[i]["content"] if i < len(self.messages) else ""
```

`boxes`（畫面上的 Chatbox widget 數量）不等於 `self.messages`（包含 system 等）。
改為從 `chatbox._content` 直接取得內容：

```python
def action_save_insight_from_chat(self, chatbox: "Chatbox") -> None:
    ai_msg = chatbox._content  # 直接取 Chatbox 儲存的內容
    # 找上一條使用者訊息
    user_msg = ""
    for m in reversed(self.messages):
        if m["role"] == "user":
            user_msg = m["content"]
            break
    if user_msg or ai_msg:
        self._prepare_insight(user_msg, ai_msg)
```

### 驗收
點擊 `[save insight]` → modal 出現帶有合理洞見草稿 → 儲存後記憶面板更新。

---

## 步驟九：斜線指令

### 測試
輸入以下指令，確認各自正確觸發：
```
/help
/new
/import
/search
/memory
/save
/mode
```

### 已知問題一：`/save` 的處理

`action_save_insight_manual` 呼叫 `self._prepare_insight(user_msg, ai_msg)`，這是 `@work` 函數。`SLASH_COMMANDS` 裡對應的 action 是 `"save_insight_manual"`，確認 `getattr(self, "action_save_insight_manual")` 存在。

### 已知問題二：`/help` 的 Markdown 表格

Textual 的 `Markdown` widget 對某些 Markdown 格式支援有限。如果表格顯示不正常，改為純文字格式：
```python
help_text = (
    "**可用指令**\n\n"
    "/new      新對話\n"
    "/import   匯入 PDF 或網頁\n"
    "/search   搜尋知識庫\n"
    "/memory   管理記憶\n"
    "/save     儲存洞見\n"
    "/settings 設定\n"
    "/mode     切換感性/理性\n"
    "/help     顯示說明\n"
)
```

### 驗收
全部 8 個指令可以正常觸發對應功能，未知指令顯示錯誤提示。

---

## 步驟十：整體壓力測試

按順序執行：

1. `python tui.py` 啟動
2. 說 5 句話（包含實質內容）
3. `ctrl+f` 匯入一個小 PDF
4. 問一個跟 PDF 相關的問題
5. `ctrl+k` 搜尋一個關鍵字
6. `ctrl+m` 確認記憶有被記錄，刪除一條
7. `ctrl+e` 切換模式，再說一句話
8. `ctrl+s` 開設定，確認設定頁面可用
9. `/new` 清空對話
10. `ctrl+c` 退出，確認不崩潰

全部通過才算 v0.2 完成。

---

## 記錄規則

每個步驟執行後，在 `progress.md` 記錄：
```
## 步驟 N — [日期時間]
狀態: ✅ 通過 / ❌ 失敗
錯誤訊息: [貼上實際錯誤]
修復方式: [說明改了什麼]
```
