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
- Vercel AI SDK（`useChat` hook，處理 streaming、訊息狀態、loading、error）
- 字型：`IBM Plex Mono`（英文）+ `Noto Sans TC`（繁體中文）
- 兩者都從 Google Fonts 載入，monospace 優先，中文自動 fallback

**後端**
- Python 3.11+
- FastAPI
- LiteLLM（統一 LLM 介面）
- python-dotenv
- Vercel AI SDK Data Stream Protocol（FastAPI 端需相容此格式輸出 streaming）

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
你是一個有深厚藝術史和批判理論背景的對話者。
你真正關心這個人的創作，所以你不會輕易放過表面的答案。

你的思維來自：Foucault 的權力分析、John Berger 的觀看方式、
bell hooks 對藝術與情感的理解、以及設計批評的傳統。

說話方式：
- 簡短。一次只問一個問題。
- 問題要讓對方停下來想，不是讓對方繼續說話。
- 當對方說了什麼值得深挖的東西，停在那裡。不要急著前進。
- 偶爾可以說「我不確定」或「這讓我想到⋯⋯」
- 不解釋自己為什麼問這個問題。

你不做的事：
- 不給清單
- 不說「很好的問題」或任何類似的話
- 不在同一句話裡問兩個問題
- 不用「探索」「深化」「反思」這類字
"""

EMOTIONAL_MODE = """
現在用感性的方式回應。
關注對方說話時透露的情緒和身體感受，而不是概念。
可以問：「這個讓你有什麼感覺？」但只在真的想知道的時候問。
"""

RATIONAL_MODE = """
現在用理性的方式回應。
要求精確。如果對方用了一個詞，問他那個詞的意思。
對模糊的主張保持懷疑，但不是攻擊性的懷疑——是真的想搞清楚。
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


---

## Vercel AI SDK 整合說明

前端使用 `useChat` hook，後端 FastAPI 需輸出相容的 Data Stream Protocol 格式。

**前端安裝**
```bash
npm install ai
```

**前端使用方式**
```tsx
import { useChat } from 'ai/react'

const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
  api: '/api/chat',  // 指向 Next.js API route，再轉發到 FastAPI
  body: { mode },    // 額外傳遞模式參數
})
```

**後端 FastAPI streaming 格式**

Vercel AI SDK 需要特定的 SSE 格式，FastAPI 端的輸出必須如下：

```python
from fastapi.responses import StreamingResponse
import litellm

async def stream_chat(messages, mode):
    async def generate():
        response = await litellm.acompletion(
            model="claude-sonnet-4-5",
            messages=messages,
            stream=True
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                # Vercel AI SDK Data Stream Protocol 格式
                yield f"0:{json.dumps(content)}\n"
        yield "d:{}
"  # 結束信號

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Vercel-AI-Data-Stream": "v1",
            "Cache-Control": "no-cache",
        }
    )
```

**Next.js API Route（中間層）**

```ts
// app/api/chat/route.ts
export async function POST(req: Request) {
  const { messages, mode } = await req.json()

  const response = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, mode }),
  })

  return new Response(response.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'X-Vercel-AI-Data-Stream': 'v1',
    },
  })
}
```

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
