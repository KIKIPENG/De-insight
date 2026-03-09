# De-insight

給視覺創作者與設計工作者使用的終端機 AI 思考夥伴。

它不只是整理重點，而是會追問你還沒說清楚的部分，幫你把想法長出結構。

---

## 安裝

```bash
curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash
```

然後執行：

```bash
de-insight
```

更新 / 解除安裝：

```bash
de-insight --update
de-insight --uninstall
```

> 需要 **Python 3.11+**、**git**、**macOS / Linux**。

---

## 功能（v0.8）

- **策展式對話**：感性 / 理性模式切換，互動式提問（`<<SELECT>>`、`<<CONFIRM>>` 等）
- **記憶系統**：從對話抽取洞見，需使用者確認後才儲存
- **知識庫**：支援 PDF / URL / DOI 匯入，建構知識圖譜（LightRAG），對話中自動引用
- **專案隔離**：每個專案各自管理對話、記憶、知識與圖片
- **本地 GGUF Embedding**：jina-embeddings-v4（Q4_K_M, 1024 維）透過 llama-server，首次啟動自動安裝
- **圖片庫**：Web 上傳 / 搜尋 / 選取，多模態 embedding，TUI 可用 `@mention` 帶圖
- **匯入管線**：背景 job queue，含速率限制、重試、匯入後驗證
- **多模型支援**：OpenAI、Anthropic、DeepSeek、OpenRouter、Google AI Studio、MiniMax

已知限制：
- TUI 目前不支援 inline image rendering。

---

## 設定

編輯 `~/.deinsight/app/.env`（或在 TUI 中按 `Ctrl+S`）：

| 變數 | 說明 | 範例 |
|------|------|------|
| `LLM_MODEL` | 聊天模型 | `openai/gpt-4o` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `OPENAI_API_BASE` | 自訂 API 端點（OpenRouter、MiniMax 等） | `https://openrouter.ai/api/v1` |
| `RAG_LLM_MODEL` | 知識圖譜用的獨立模型（選填） | `openai/gpt-4o-mini` |
| `DEINSIGHT_HOME` | 自訂資料根目錄（預設 `~/.deinsight`） | |
| `GGUF_AUTO_INSTALL` | 首次啟動自動安裝 GGUF（預設 `1`） | |

所有 LLM 均走雲端 API，Embedding 為本地 GGUF。

---

## 快捷鍵

| 按鍵 | 功能 |
|------|------|
| `Ctrl+S` | 設定 |
| `Ctrl+E` | 切換感性 / 理性模式 |
| `Ctrl+N` | 新對話 |
| `Ctrl+P` | 專案管理 |
| `Ctrl+F` | 匯入 PDF / URL |
| `Ctrl+K` | 搜尋知識庫 |
| `Ctrl+M` | 記憶管理 |
| `Ctrl+G` | 記憶關係圖 |
| `Ctrl+L` | 開啟圖片庫（瀏覽器） |
| `Ctrl+D` | 文件管理 |
| `Ctrl+B` | 批次匯入 |

在聊天輸入框打 `/help` 可查看完整指令列表。

---

## 測試

```bash
~/.deinsight/app/.venv/bin/python -m pytest -q tests/
```

---

## 疑難排解

### GGUF embedding 編譯失敗

需要 Xcode Command Line Tools（macOS）或 cmake + C++ 編譯器（Linux）：

```bash
xcode-select --install   # macOS
sudo apt install cmake build-essential   # Ubuntu/Debian
```

### 後端連線失敗

後端會隨 TUI 自動啟動。若失敗，檢查 port 8000：

```bash
curl -m 3 -sS http://127.0.0.1:8000/api/health
```

---

## 資料位置

```
~/.deinsight/
├── app/              # 原始碼 + venv（install.sh 安裝）
├── v0.7/             # 使用者資料
│   ├── app.db
│   └── projects/{id}/
│       ├── memories.db
│       ├── conversations.db
│       ├── lancedb/
│       ├── lightrag/
│       ├── images/
│       └── documents/
└── gguf/             # Embedding 模型 + llama-server
    ├── llama.cpp/
    ├── models/
    └── logs/
```

---

## 貢獻

1. Fork → 開分支
2. 修改前先讀相關檔案
3. 跑 `python -m pytest tests/` 確認沒壞
4. 發 PR

---

## License

TODO（尚未定義）
