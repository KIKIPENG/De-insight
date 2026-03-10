<h4 align="center">
  <a href="https://github.com/KIKIPENG/De-insight/blob/main/README.md">English</a> | <strong>繁體中文</strong>
</h4>

<h1 align="center">◈ De-insight</h1>

<p align="center">
  給視覺創作者的 AI 思考夥伴。<br>
  記得你說過什麼、看過你收集的圖片、在你的想法演變時告訴你。
</p>

<pre align="center">curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash</pre>

---

## 這是什麼

創作者在發展論述的過程裡，腦中有大量閱讀、觀察、直覺反應，但要把這些東西組織成一份站得住腳的文字，中間有一段很長的路。

De-insight 試圖縮短這段路。

它不是搜尋引擎，不是寫作助手，也不是通用聊天機器人。它更接近一個長期合作的策展人——記得你上次說了什麼、讀過你餵給它的文獻、看過你收集的參考圖片，在你說得模糊的時候追問你到底什麼意思。

當你的想法在三個月裡悄悄變了，它會告訴你。當你嘴上說喜歡極簡，但收集的圖片全是粗糙手工的東西，它會指出來。

準備好要寫論述了，它幫你從自己的語言開始組織，理論放最後當支撐，不當主角。

> 這是一個為了畢業製作發想而開發的軟體。為一個人的工作流程做的，但開放給所有想法接近的人。

---

## 它做的事

### 對話

兩種模式。**感性模式**關注你選詞背後的猶豫和偏好，不急著分析；**理性模式**要求精確，一個詞含糊就問你那個詞是什麼意思。

策展人說話直接、簡短、有立場。它不是中立的——你問它怎麼看，它會說。

當你說「我想寫論述了」，它走三步：提煉命題、從知識庫找證據、起草文字。每一步等你確認才往下。

### 知識庫

把你讀的東西餵進來：PDF、網頁連結、DOI、arXiv、純文字。系統在背景建知識圖譜，你在 MenuBar 上能看到進度。

對話時自動撈相關段落注入 context。策展人只能用知識庫裡有的東西——沒有的理論家不能捏造，證據不足要直接說。

兩種檢索模式：**快速**（向量搜尋，語義匹配）和**深度**（圖譜推理，能找到跨文獻的概念連結——比如 Foucault 的「全景敞視」和 Berger 的「觀看之道」之間的結構性關聯）。

### 記憶

每次對話結束，系統分析你說過的話，抽出值得留下的東西——洞見、問題、對作品的反應。候選記憶要你確認了才存。

記憶帶主題標籤和面向分類。下次對話時策展人帶著這些記憶跟你說話。

**思維演變偵測**：當你存入一條新洞見，系統比對過去的洞見，偵測觀點的演變或矛盾。「你三個月前認為 X，但現在你說的暗示 Y。」

### 圖片

在瀏覽器開圖片庫頁面，上傳你的參考圖片。每張圖自動生成三段式分析：

- **CONTENT**：物件辨識——書名、作者，以及內容和設計語言之間的關係
- **STYLE_TAGS**：風格座標——製作態度、密度節奏、色彩傾向、時代感、文化軸、態度
- **DESCRIPTION**：策展語彙描述

**視覺偏好萃取**：收集 5 張以上圖片後，系統自動歸納你的風格傾向。結果注入對話，策展人能說出「你的收藏偏向粗糙、手工、實驗性的東西」。

**交叉偵測**：當你的圖片偏好和文字洞見矛盾，系統會指出來——「你說你喜歡極簡，但你收集的圖片全是另一回事。」

### 專案

每個專案有自己的對話、記憶、知識庫和圖片。用來分隔不同的創作脈絡。

---

## 安裝

```bash
curl -fsSL https://raw.githubusercontent.com/KIKIPENG/De-insight/main/install.sh | bash
```

```bash
de-insight           # 啟動
de-insight --update  # 更新
de-insight --uninstall  # 移除
```

第一次啟動會引導你選 LLM provider。Embedding 模型（約 1.5 GB）在第一次用到時自動下載編譯。

需要 **Python 3.11+**、**git**、**macOS / Linux**。

---

## 設定

編輯 `~/.deinsight/app/.env`，或在 TUI 按 `Ctrl+S`。

```
# 主聊天模型
LLM_MODEL=openai/deepseek/deepseek-chat-v3-0324
OPENROUTER_API_KEY=<key>
OPENAI_API_BASE=https://openrouter.ai/api/v1

# 知識庫建圖 + 圖片分析（建議 Gemini，免費且穩定）
RAG_LLM_MODEL=gemini-2.5-flash
VISION_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=<key>
```

Embedding 完全本地，不需要 API key。

---

## 系統需求

| 項目 | 最低 | 建議 |
|------|------|------|
| 晶片 | 任何 | Apple M1+（Metal 加速） |
| 記憶體 | 8 GB | 16 GB |
| 磁碟 | 4 GB | 10 GB+ |
| Python | 3.11+ | — |

macOS 需要 `xcode-select --install`，Linux 需要 `cmake` + `build-essential`。

---

## 技術架構

```
tui.py → app.py（Textual TUI + FastAPI 後端）

聊天      → DeepSeek V3 via OpenRouter
知識圖譜  → Gemini via Google AI Studio
圖片描述  → Gemini via Google AI Studio
Embedding → jina-embeddings-v4 GGUF via llama-server（本地）

知識庫    → LightRAG（JSON/NetworkX）
向量索引  → LanceDB
記憶      → SQLite（aiosqlite）
圖片庫    → LanceDB + Web 前端
```

```
~/.deinsight/
├── app/                  原始碼 + venv
├── v0.7/projects/{id}/   專案資料
│   ├── memories.db       洞見、問題、反應、偏好
│   ├── conversations.db  對話歷史
│   ├── lancedb/          向量索引
│   ├── lightrag/         知識圖譜
│   ├── images/           圖片
│   └── documents/        PDF
└── gguf/                 Embedding 模型 + llama-server
```

---

## 疑難排解

**Embedding 編譯失敗**：macOS 跑 `xcode-select --install`，Linux 跑 `sudo apt install cmake build-essential`。

**後端連不上**：後端隨 TUI 自動啟動。`curl -m 3 -sS http://127.0.0.1:8000/api/health` 確認。

**圖片庫打不開**：確認 TUI 正在跑，然後開 `http://localhost:8000/gallery`。

---

## 已知限制

- 終端機不能 inline 顯示圖片。圖片功能透過描述文字和語意檢索運作。
- 第一次啟動要編譯 llama.cpp 和下載模型，需要幾分鐘。
- 策展人全程講繁體中文。

---

## 測試

```bash
~/.deinsight/app/.venv/bin/python -m pytest -q tests/
```

---

## License

尚未定義。

---

<sub>made by KIKI PENG — 藝術設計畢業製作</sub>
