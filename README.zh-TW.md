# De-insight

給視覺創作者與設計工作者使用的終端機 AI 思考夥伴。

它不只是整理重點，而是會追問你還沒說清楚的部分，幫你把想法長出結構。

---

## 功能（v0.7）

- 策展式對話：感性 / 理性模式切換
- 記憶系統：從對話抽取洞見，需使用者確認後才儲存
- 知識庫：支援 PDF / URL 匯入，對話中自動引用
- 專案隔離：每個專案各自管理對話、記憶、知識與圖片
- 本地向量：jina-clip-v1（dim=512），支援文字與圖片語意檢索
- 圖片庫：Web 上傳 / 搜尋 / 選取，TUI 可用 `@mention` 帶圖
- 首次啟動 Onboarding：引導設定模型與 embedding 模式

已知限制：
- TUI 目前不支援 inline image rendering。

---

## 需求

- Python 3.10+
- macOS / Linux（Windows 未完整測試）
- 若使用本地 embedding（`EMBED_MODE=local`）需額外模型空間（約 1GB）

---

## 快速開始

```bash
git clone <repo-url>
cd De-insight

cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..

echo 'LLM_MODEL=anthropic/claude-sonnet-4-20250514' > .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env

source backend/.venv/bin/activate
cd backend && uvicorn main:app --reload &
cd ..

python3 tui.py
```

---

## 設定

主要設定寫在專案根目錄 `.env`：

- `LLM_MODEL`
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` / `DEEPSEEK_API_KEY` / `MINIMAX_API_KEY` / `CODEX_API_KEY`
- `EMBED_MODE`（`local` 或 API）
- `EMBED_PROVIDER`
- `EMBED_DIM`
- `OPENAI_API_BASE`（選填）
- `EMBED_API_BASE`（選填）
- `RAG_LLM_MODEL`（選填）
- `DEINSIGHT_HOME`（選填，自訂資料根目錄）

---

## 執行

後端：

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload
```

TUI：

```bash
source backend/.venv/bin/activate
python3 tui.py
```

Gallery：
- `http://localhost:8000/gallery`

---

## 測試

```bash
source backend/.venv/bin/activate

python3 -m unittest tests/test_conversation_isolation.py -v
python3 -m unittest tests/test_prompt_parser.py -v
python3 -m unittest tests/test_rag_switch.py -v
```

註：`test_rag_switch.py` 需要 `lightrag` 與對應環境依賴。

---

## 資料與相容性

- 使用者資料預設在 `~/.deinsight/v0.6/`
- v0.6 → v0.7 不相容
- 不提供自動 migration

---

## 貢獻

1. 開分支
2. 修改後先跑 compile / 測試
3. 發 PR

---

## License

TODO（尚未定義）

