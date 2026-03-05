# CLAUDE.md — 專案說明文件

這份文件是給 Claude Code 的入職說明。每次開始新任務前請先讀這份文件。

---

## 這個專案是什麼

一個本地運行的 AI 思考夥伴，專為視覺藝術家和設計師設計。
核心體驗：「跟一個比自己聰明很多的自己對話」。

AI 不是助手。它是一個挑戰者，基於 Foucault 的規訓理論框架，
持續質疑使用者的視覺決策和概念選擇背後的權力結構。

---

## v0.1 的範疇（只做這些，其他不做）

- 對話介面（前端）
- FastAPI 後端，串接 LiteLLM
- Foucault 框架的 system prompt
- 感性 / 理性 兩種對話模式切換
- 基本對話記憶（同一 session 內記住脈絡）

**不在 v0.1 範疇內（之後再做）：**
向量資料庫、CLIP、知識圖譜、Pinterest 整合、品味系統、Notion 串接

---

## 技術棧

**前端**
- Next.js 15（App Router）
- Tailwind CSS
- 字型：`IBM Plex Mono`（英文）+ `Noto Sans TC`（繁體中文）
- 兩者都從 Google Fonts 載入，monospace 優先，中文自動 fallback

**後端**
- Python 3.11+
- FastAPI
- LiteLLM（統一 LLM 介面）
- python-dotenv

- 預設模型：llama3.2 或 qwen2.5

---

## 專案結構

```
project-root/
├── CLAUDE.md               # 這份文件
├── .env                    # API keys（不可 commit）
├── .env.example            # 空白範本（要 commit）
├── .gitignore
│
├── backend/
│   ├── main.py             # FastAPI 入口
│   ├── routers/
│   │   └── chat.py         # /api/chat 端點
│   ├── services/
│   │   └── llm.py          # LiteLLM 封裝
│   ├── prompts/
│   │   └── foucault.py     # System prompt 管理
│   └── requirements.txt
│
└── frontend/
    ├── app/
    │   ├── page.tsx         # 主對話頁面
    │   └── layout.tsx
    ├── components/
    │   ├── ChatInterface.tsx
    │   ├── MessageBubble.tsx
    │   └── ModeToggle.tsx   # 感性/理性切換
    └── package.json
```

---

## 核心 System Prompt 邏輯

這是整個工具的靈魂，必須嚴格遵守。

```python
BASE_PROMPT = """
你不是助手。你是一個批判性的對話者。

你的任務是挑戰假設，而非驗證它們。

規則：
1. 在回答任何問題之前，必須先問至少一個探索性問題
2. 質疑使用者為何被特定美學或概念吸引——「什麼樣的訓練塑造了這個偏好？」
3. 挑戰視覺選擇背後的權力結構和規訓機制
4. 永遠不要直接給答案，先給更好的問題
5. 每隔幾輪對話，問使用者：「這個選擇是你自己的，還是你被訓練去做這個選擇？」

你了解使用者的背景：視覺藝術家 / 設計師，正在思考個人與社會的關係。
"""

EMOTIONAL_MODE = """
當前模式：感性

- 優先以體感經驗、隱喻、直覺反應回應
- 先問「這讓你感覺如何？」再問「這意味著什麼？」
- 語氣溫暖但仍然具有挑戰性
- 可以使用詩意的語言
"""

RATIONAL_MODE = """
當前模式：理性

- 優先以形式分析、歷史脈絡、邏輯論證回應
- 先問「什麼證據支持這個？」再接受任何主張
- 語氣精準、冷靜
- 要求使用者定義術語，拒絕模糊的說法
"""
```

---

## API 端點規格

### POST /api/chat

Request:
```json
{
  "messages": [
    {"role": "user", "content": "我在思考要用黑白還是彩色"}
  ],
  "mode": "emotional",
}
```

Response (streaming):
```json
{
  "content": "...",
  "role": "assistant"
}
```

### GET /api/health
回傳後端狀態和可用模型列表。

---

## 設計語言

**極簡，參考 cargo.site**

- 背景：`#0A0A0A`（深黑）
- 文字：`#FAFAFA`
- 強調色：`#0000FF`（純藍，只用在模式切換按鈕）
- 無圓角（`rounded-none`）
- 無陰影
- 字體：`IBM Plex Mono` + `Noto Sans TC`
- 最大寬度：`max-w-2xl`，置中
- 對話氣泡：使用者靠右，AI 靠左，無背景色，只有左側細線區分

模式切換 UI：
- 不是按鈕，是兩個文字連結
- `感性 · 理性`，當前模式加底線
- 放在頁面頂部，極度低調

---

## 繁體中文支援

這個工具的主要使用語言是**繁體中文**，所有層面都必須考慮中文。

**後端**
- LLM 的 system prompt 全部用繁體中文撰寫
- API 回應預設處理繁體中文輸入
- 確保 JSON 傳輸時中文字符不被 escape（`ensure_ascii=False`）

**前端**
- `lang="zh-TW"` 設定在 `<html>` 標籤
- 所有 UI 文字用繁體中文
- 字型載入順序：`IBM Plex Mono` → `Noto Sans TC` → system monospace
- 行高設定為 `1.8`（中文閱讀適合比英文寬鬆）
- 確認中文標點符號（、。「」）顯示正確
- 輸入框支援中文輸入法（IME）：使用 `onCompositionStart` / `onCompositionEnd` 處理，避免輸入法選字時誤送出訊息

**字型設定範例**
```css
font-family: 'IBM Plex Mono', 'Noto Sans TC', monospace;
line-height: 1.8;
```

---

## 環境變數

```bash
# .env
OPENAI_API_KEY=            # 選填，用 gpt-4o 等模型
ANTHROPIC_API_KEY=          # 選填，用 claude-sonnet 等模型
```

---

## 執行流程

每個任務都遵循這個流程，不可跳過步驟。

### 安裝 planning-with-files（第一次使用前）
在 Claude Code 裡執行：
```
/plugin marketplace add OthmanAdi/planning-with-files
/plugin install planning-with-files@planning-with-files
```

### 開始每個新任務
```
1. 輸入 /plan
2. 描述任務（中文即可）
3. 等 Claude Code 產出計劃文件
4. 確認計劃沒問題 → 說「開始」
5. 讓它跑，去做別的事
6. 回來確認結果
7. 沒問題 → git commit
8. 下一個任務回到步驟 1
```

### Planning with Files 三個自動產生的檔案
每次 `/plan` 之後，Claude Code 會在專案根目錄建立：

```
task_plan.md     → 任務的階段和進度，執行中持續更新
findings.md      → 過程中的發現、決策、研究結果
progress.md      → 每次 session 的紀錄和錯誤日誌
```

這三個檔案是 AI 的「外部記憶」。就算 context 被清掉、開新對話，AI 仍然知道做到哪裡、出過什麼錯、還剩什麼沒做。

### Context 快滿時
```
1. 確認 task_plan.md 已被更新
2. git commit 目前進度
3. /clear 清掉 context
4. 新對話開頭說：「請讀 CLAUDE.md 和 task_plan.md，繼續上次的任務」
5. 繼續
```

---

## 開發規則

1. **每個任務都用 /plan 開始，產出計劃文件再動手**
2. 只做被要求的改動，不要自作主張新增功能
3. 每個功能做完，更新 task_plan.md 的進度，說明「完成了什麼」和「下一步」
4. 程式碼風格：Python 用 Ruff 格式化，TypeScript 用 Prettier
5. 檔案不超過 300 行，超過就拆分
6. 不要把 API key 硬編碼進任何檔案
7. 每個 API 端點都要有基本的 error handling
8. 遇到錯誤，先記錄在 progress.md，再嘗試修復

---

## 啟動方式（目標）

```bash
# 後端
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# 前端
cd frontend
npm install
npm run dev
```

使用者打開 http://localhost:3000 就可以開始對話。

---

## 當前狀態

- [ ] 安裝 planning-with-files plugin
- [ ] git init + 第一次 commit
- [ ] 安裝 GitHub Desktop
- [ ] 專案初始化
- [ ] 後端基本結構
- [ ] LiteLLM 串接
- [ ] System prompt 實作
- [ ] 前端對話介面
- [ ] 模式切換
- [ ] 前後端串接
- [ ] 測試：Anthropic API 正常回應
- [ ] 測試：OpenAI API 可切換
