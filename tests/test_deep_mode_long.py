"""Deep mode stress test — 10 long-form mixed-concept passages.

每篇 KB 文章 500+ 字，混雜多重概念（不是乾淨的單一論點），
測試 De-insight 能否在「噪音」中精準抓到正確的結構脈絡。

測試流程：
  Phase 1: 把 10 篇長文各抽出 2-3 個 claims，全部灌入同一個 mega store
  Phase 2: 10 個使用者對話場景，每個場景用 structural search 找匹配
  Phase 3: Precision/Recall 量化 — 期望命中的有沒有找到？不該找到的有沒有誤判？
  Phase 4: Retriever 端對端 — _retrieve_from_claims 的雙路由能力
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.schemas import Claim, SourceKind, RetrievalPlan
from core.thought_extractor import ThoughtExtractor, LLMCallable
from core.stores.claim_store import ClaimStore
from core.retriever import Retriever


def _llm(resp):
    async def f(prompt): return resp
    return LLMCallable(func=f)

def _store(pid="mega"):
    return ClaimStore(project_id=pid, db_path=Path(tempfile.mktemp(suffix=".db")))


# ═══════════════════════════════════════════════════════════════════════
# 10 篇長文 KB 文章 — 每篇混雜多重概念
# ═══════════════════════════════════════════════════════════════════════

KB_ARTICLES = {

# ──────────────────────────────────────────────────────────────────────
# KB-1: 日本茶道（混雜：約束→本質 + 時間倫理 + 留白 + 材料誠實）
# ──────────────────────────────────────────────────────────────────────
"kb_chado": {
    "text": """
日本茶道的核心不在於泡茶的技法，而在於整個儀式所建構的感知框架。千利休將茶室
縮小到僅僅四疊半榻榻米的空間——這個極端的空間約束迫使主客之間的距離壓縮到
無法迴避的程度，讓每一個細微的手勢和聲音都被放大。在這種限制下，裝飾性的社交
語言被剝除，只留下最核心的人與人之間的「間」——那個沉默但充滿張力的空隙。

同時，茶道對時間的處理方式深具啟示。一期一會的觀念意味著每次茶會都是不可重複
的事件，這種「一次性」不是效率的敵人，而是意義的來源。水在鐵壺中沸騰的等待、
茶筅攪拌的節奏、季節性花材的選擇——這些都是時間作為材料被織入體驗的方式。
等待不是浪費，而是體驗的結構性成分。

茶碗的選擇更是直接表達了物質的倫理：樂燒茶碗故意留下手捏的痕跡和窯燒的偶然
變色，拒絕機器的均質完美。每一個不規則都在說：這是特定的手、在特定的時間、
與特定的火之間的一次對話。材料不被掩飾，而是被慶祝。

最後，茶室中的掛軸和花只放一幅、一枝——不是因為窮，而是因為空間的空是主動的。
那片空牆不是「沒有掛畫」，而是「選擇了不掛」，是一個邀請觀者用自己的感知去
填充的留白。空不是缺席，而是最大的在場。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "茶道的極端空間約束迫使社交裝飾被剝除，只留下人與人之間不可化約的核心互動",
                "critique_target": ["裝飾性社交", "空間浪費"],
                "value_axes": ["本質", "約束", "間"],
                "materiality_axes": ["四疊半空間", "榻榻米"],
                "labor_time_axes": [],
                "abstract_patterns": ["約束→剝除→本質浮現", "限制→放大感知"],
                "theory_hints": ["千利休", "侘寂", "間"]
            },
            {
                "core_claim": "茶道中的等待與一次性不是效率的敵人而是意義的來源，時間作為材料被織入體驗",
                "critique_target": ["效率崇拜", "可重複性"],
                "value_axes": ["時間價值", "一次性", "不可壓縮"],
                "materiality_axes": ["鐵壺", "茶筅", "季節花材"],
                "labor_time_axes": ["等待", "沸騰節奏", "一期一會"],
                "abstract_patterns": ["時間→材料化→意義", "不可壓縮→價值"],
                "theory_hints": ["一期一會"]
            },
            {
                "core_claim": "樂燒茶碗故意保留手捏痕跡與窯燒偶然，拒絕均質完美，材料不被掩飾而被慶祝",
                "critique_target": ["機器均質", "完美主義"],
                "value_axes": ["材料誠實", "不完美", "痕跡"],
                "materiality_axes": ["陶土", "窯燒", "手捏痕跡"],
                "labor_time_axes": ["手作時間"],
                "abstract_patterns": ["媒介→忠於自身→正當性", "痕跡→誠實"],
                "theory_hints": ["侘寂", "樂燒"]
            },
            {
                "core_claim": "茶室中只掛一幅畫、只放一枝花，空牆不是缺少而是主動選擇的留白，空不是缺席而是最大的在場",
                "critique_target": ["填滿恐懼", "裝飾過度"],
                "value_axes": ["缺席即在場", "留白", "空即容器"],
                "materiality_axes": ["空牆", "掛軸", "花"],
                "labor_time_axes": [],
                "abstract_patterns": ["缺席→主動在場", "空→揭示被掩蓋之物", "減法→意義放大"],
                "theory_hints": ["侘寂", "千利休", "間"]
            }
        ],
        "thought_summary": "茶道作為約束、時間、材料誠實與留白四重交織的感知框架",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-2: 嘻哈取樣文化（混雜：脈絡決定意義 + 匱乏→創新 + 收編消解抵抗）
# ──────────────────────────────────────────────────────────────────────
"kb_hiphop_sampling": {
    "text": """
嘻哈取樣的核心操作是「去脈絡化再脈絡化」。DJ Premier 從 1960 年代靈魂樂的
某一小節中截取兩秒的鋼琴和弦，放進一個關於布魯克林街頭生存的 beat 裡——
同樣的聲音碎片，在新的語境中產生了完全不同的情感意義。這不是借用或致敬，
而是一種激進的意義生產：脈絡就是意義，改變脈絡就是創造。

這種創作方式誕生於極端的物質匱乏。南布朗克斯的年輕人沒有樂器、沒有錄音室、
沒有音樂學院的訓練——他們有的只是父母那一代留下的唱片和一台混音器。但正是
這種匱乏迫使他們發明了一種全新的音樂生產方法：不是創作全新的聲音，而是重組
既有的聲音碎片。被迫的限制催生了一套美學，這套美學後來改變了全世界的音樂生產。

然而，當嘻哈從地下進入主流，取樣從反叛變成了產品。唱片公司用同樣的取樣技法
製造流水線式的商業歌曲，街頭的即興混音被標準化為 Pro Tools 裡的工作流程。
符號的外殼保留了，但抵抗的核心被掏空。Nike 請饒舌歌手賣球鞋的那一刻，
嘻哈的反建制能量就被收編進了它原本批判的系統裡。可見性和商業成功恰恰是
批判力量消解的起點。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "嘻哈取樣透過去脈絡化再脈絡化產生全新意義，脈絡是意義的構成條件而非容器",
                "critique_target": ["原創性迷思", "固定意義"],
                "value_axes": ["再脈絡化", "意義生產", "脈絡即意義"],
                "materiality_axes": ["唱片", "混音器", "聲音碎片"],
                "labor_time_axes": ["即興混音"],
                "abstract_patterns": ["脈絡→決定意義", "去脈絡→再脈絡→新意義"],
                "theory_hints": ["Duchamp現成物", "後現代拼貼"]
            },
            {
                "core_claim": "嘻哈誕生於物質匱乏，被迫的限制催生了全新的音樂生產方法，匱乏是強迫創新的機制",
                "critique_target": ["資源決定論", "正規教育壟斷"],
                "value_axes": ["匱乏創新", "被迫發明"],
                "materiality_axes": ["二手唱片", "混音器"],
                "labor_time_axes": ["即興", "非正式傳承"],
                "abstract_patterns": ["匱乏→被迫創新→新方法", "限制→新美學"],
                "theory_hints": ["bricolage", "情境主義"]
            },
            {
                "core_claim": "嘻哈進入主流後取樣從反叛變成商品，符號外殼保留但抵抗核心被掏空，可見性即消解起點",
                "critique_target": ["商業收編", "主流化"],
                "value_axes": ["收編消解抵抗", "可見性悖論"],
                "materiality_axes": ["商品", "品牌"],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→被收編→失去力量", "可見性→消解", "成功→自毀"],
                "theory_hints": ["Adorno文化工業", "Hebdige次文化"]
            }
        ],
        "thought_summary": "嘻哈取樣展現脈絡決定意義、匱乏催生創新、以及收編消解抵抗三重結構",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-3: 清水模建築（混雜：媒介誠實 + 約束→本質 + 時間痕跡）
# ──────────────────────────────────────────────────────────────────────
"kb_concrete": {
    "text": """
安藤忠雄的清水模建築之所以震撼人心，不在於它的形式語言，而在於它所體現的
倫理立場。當混凝土的毛孔、氣泡、模板接縫全部暴露在外，建築就在宣告：
我不需要用大理石貼面來假裝自己是別的東西。這是一種媒介的自我揭示——混凝土
承認自己是混凝土，正如 Greenberg 要求繪畫承認畫布的平面性。

但清水模的倫理不止於此。模板的拆除痕跡、澆灌時的振搗紋路、甚至偶爾出現的
色差——這些都是時間和勞動留在材料上的證據。每一面牆都記錄了一個特定的下午、
特定的工人、特定的溫度和濕度。建築不只是空間的組織，它同時是時間的化石。

更微妙的是，安藤的空間處理方式暗含了「約束產生本質」的邏輯。住吉的長屋只有
不到 60 平方米，但極端的壓縮迫使居住者重新定義「什麼是必要的」。去掉走廊、
去掉多餘的房間、去掉裝飾——剩下的是光、風、和身體在空間中移動的最基本體驗。
限制不是貧困的表現，而是一種有意識的設計策略：透過拿掉，讓留下的更清晰。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "清水模建築讓混凝土暴露所有製程痕跡，是媒介的自我揭示，拒絕偽裝成別的材料",
                "critique_target": ["材料偽裝", "貼面文化"],
                "value_axes": ["媒介誠實", "自我揭示", "物質忠誠"],
                "materiality_axes": ["混凝土", "毛孔", "氣泡", "模板接縫"],
                "labor_time_axes": [],
                "abstract_patterns": ["媒介→忠於自身→正當性", "拒絕偽裝→誠實"],
                "theory_hints": ["安藤忠雄", "Greenberg", "媒介特殊性"]
            },
            {
                "core_claim": "清水模牆面的模板痕跡和色差是特定時間、工人、環境條件的化石，建築即時間紀錄",
                "critique_target": ["均質完美", "去時間化"],
                "value_axes": ["時間痕跡", "勞動紀錄"],
                "materiality_axes": ["模板紋路", "色差", "振搗痕跡"],
                "labor_time_axes": ["特定下午", "特定工人", "澆灌過程"],
                "abstract_patterns": ["時間→凝固於成品", "痕跡→歷史紀錄"],
                "theory_hints": ["物質文化", "痕跡學"]
            },
            {
                "core_claim": "住吉的長屋用極端空間壓縮迫使居住者重新定義必要性，透過拿掉讓留下的更清晰",
                "critique_target": ["空間浪費", "裝飾性空間"],
                "value_axes": ["本質", "約束", "必要性"],
                "materiality_axes": ["60平方米", "光", "風"],
                "labor_time_axes": [],
                "abstract_patterns": ["約束→剝除→本質浮現", "限制→重新定義必要"],
                "theory_hints": ["最小住宅", "空間現象學"]
            }
        ],
        "thought_summary": "清水模建築交織媒介誠實、時間凝固與空間約束三個維度",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-4: Duchamp 與觀念藝術（混雜：脈絡決定意義 + 邊緣定義中心 + 觀看即參與）
# ──────────────────────────────────────────────────────────────────────
"kb_duchamp": {
    "text": """
Marcel Duchamp 在 1917 年把一個工廠生產的小便斗簽上「R. Mutt」送進獨立藝術
展的行為，表面上是一個挑釁，實際上是一次嚴謹的哲學實驗。這個實驗的核心問題
不是「小便斗是不是藝術」，而是「什麼條件讓一個東西成為藝術」。答案不在物件
本身的美學屬性中，而在於它被放置的脈絡——美術館的白牆、展覽的框架、藝術體制
的認可系統。同一個物件，在五金行是商品，在美術館是藝術。脈絡不是容器，
而是意義的共同生產者。

但 Duchamp 的動作同時也是一次關於「邊界」的實驗。他選擇了一個最不可能被視為
藝術的東西——一個尿斗——正是為了測試藝術這個範疇的極限。如果連小便斗都可以
是藝術，那麼「藝術」的定義就不能依靠任何內在屬性，只能依靠外部的制度性框架。
邊緣案例不是例外，而是最能揭示規則本質的地方。

更深一層，Duchamp 的作品改變了觀看者的角色。在傳統藝術中，觀眾是被動的接收者；
但面對一個小便斗，觀眾被迫成為意義的主動建構者——「這是藝術嗎？」這個問題本身
就把觀眾從旁觀者變成了參與者。觀看不再是被動的接收，而是一種創造性行為。
作品的意義不在物件裡，而在物件和每個特定觀看者之間的交互中產生。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "Duchamp的現成物實驗證明意義不在物件本身而在脈絡中，脈絡是意義的共同生產者",
                "critique_target": ["本質主義美學", "內在屬性論"],
                "value_axes": ["脈絡即意義", "制度性框架"],
                "materiality_axes": ["工業製品", "小便斗"],
                "labor_time_axes": [],
                "abstract_patterns": ["脈絡→決定意義", "改變框架→改變本質"],
                "theory_hints": ["Duchamp", "體制批判", "Danto藝術界"]
            },
            {
                "core_claim": "Duchamp選擇最不可能的物件測試藝術範疇極限，邊緣案例最能揭示規則的本質",
                "critique_target": ["內在屬性定義", "本質主義"],
                "value_axes": ["邊緣定義中心", "範疇極限"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→定義中心", "例外→揭示規則"],
                "theory_hints": ["Wittgenstein", "體制理論"]
            },
            {
                "core_claim": "面對現成物，觀眾從被動接收者變成意義的主動建構者，觀看成為創造性行為",
                "critique_target": ["被動觀看", "旁觀者角色"],
                "value_axes": ["觀看即參與", "主動建構"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["觀察→改變對象", "觀看→主動建構意義"],
                "theory_hints": ["接受美學", "Umberto Eco開放作品"]
            }
        ],
        "thought_summary": "Duchamp的現成物同時揭示脈絡決定意義、邊緣定義中心、觀看即參與三個結構",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-5: 翻譯與文化轉換（混雜：不可譯→定義本質 + 脈絡 + 觀看改變對象）
# ──────────────────────────────────────────────────────────────────────
"kb_translation": {
    "text": """
Walter Benjamin 在〈翻譯者的任務〉中提出了一個違反直覺的觀點：好的翻譯不是
讓讀者忘記自己在讀翻譯，而是讓原文中那些不可轉換的東西變得可見。翻譯的價值
恰恰在於它的「失敗」——那些翻不過去的地方，反過來定義了原文的獨特性。
「氣韻生動」翻成 "spirit resonance" 的那個落差，比任何解釋都更精確地指出了
中國美學的不可化約核心。

這跟音樂改編的邏輯驚人地相似。當一首管弦樂被改編成鋼琴獨奏，失去的不只是
音色——更是複數聲部之間的空間關係。但這個損失同時凸顯了旋律的骨架結構：
那些被合奏的豐富性所掩蓋的線條，在鋼琴上變得清晰可辨。損失不只是損失，
它同時是一種揭示。

同時，翻譯也揭示了脈絡對意義的構成作用。一個日文的「もったいない」在中文裡
可以翻成「可惜」「浪費」或「暴殄天物」，但每個翻譯都只捕捉了原詞意義場的一部分。
這意味著「もったいない」的完整意義只存在於日文的文化語境中——脈絡不是可以剝離的
外殼，而是意義不可分割的一部分。

最後，翻譯這個行為本身改變了我們對原文的理解。在翻譯之前，你以為你理解了原文；
翻譯的困難迫使你重新看待那些「理所當然」的概念。翻譯不是從 A 到 B 的搬運，
而是一面鏡子，讓 A 看到自己以前沒注意到的面向。觀看（翻譯）改變了被觀看物（原文）。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "翻譯的「失敗」恰恰定義了原文的獨特性，不可轉換之物揭示了不可化約的核心",
                "critique_target": ["透明翻譯", "等價論"],
                "value_axes": ["不可譯性", "獨特性揭示"],
                "materiality_axes": ["語言物質性"],
                "labor_time_axes": [],
                "abstract_patterns": ["轉換失敗→揭示本質", "損失→認識"],
                "theory_hints": ["Walter Benjamin", "翻譯者的任務"]
            },
            {
                "core_claim": "翻譯揭示脈絡是意義不可分割的部分，詞彙的完整意義只存在於特定文化語境",
                "critique_target": ["意義可搬運", "去脈絡理解"],
                "value_axes": ["脈絡即意義", "文化語境"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["脈絡→決定意義", "無脫脈絡的意義"],
                "theory_hints": ["語言相對論", "Sapir-Whorf"]
            },
            {
                "core_claim": "翻譯行為本身改變了對原文的理解，迫使重新看待理所當然的概念",
                "critique_target": ["翻譯=搬運", "原文理解已完成"],
                "value_axes": ["觀看改變對象", "反身性"],
                "materiality_axes": [],
                "labor_time_axes": ["翻譯勞動"],
                "abstract_patterns": ["觀察→改變對象", "翻譯→重新理解原文"],
                "theory_hints": ["詮釋學循環", "Gadamer"]
            }
        ],
        "thought_summary": "翻譯作為三重揭示裝置：揭示不可譯性、揭示脈絡構成意義、揭示觀看改變對象",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-6: 龐克與DIY文化（混雜：收編消解抵抗 + 匱乏→創新 + 媒介誠實）
# ──────────────────────────────────────────────────────────────────────
"kb_punk": {
    "text": """
龐克音樂在 1970 年代末期的倫敦和紐約幾乎同時爆發，它不只是一種音樂風格，
更是一整套關於「誰有權創作」的激進宣言。三個和弦就能組一個樂團——這不是
技術的缺陷，而是一種倫理立場：你不需要音樂學院的認證才有資格表達。技術的
匱乏不是障礙，而是解放的條件。正因為不需要精湛的技巧，任何人都可以參與，
音樂生產的門檻被徹底推倒。

DIY 美學——自己印傳單、自己錄卡帶、自己縫衣服——同時也是一種媒介誠實。
手寫的傳單不假裝自己是專業印刷品，粗糙的錄音不假裝自己是錄音室製作。
這種粗糙不是不得已的妥協，而是有意識的選擇：材料的粗糙性本身就是一種
反機構的訊息。當你用影印機翻印 zine 的時候，複印的痕跡——墨跡的不均勻、
紙張的皺褶——成為了真實性的標記。

但當大型連鎖店開始販售「做舊」風格的龐克 T-shirt，當設計公司用高解析度
印刷模仿手寫字體的「不完美」，龐克的視覺語言就被從它的生產脈絡中抽離了。
外觀被保留，但產生這個外觀的社會條件被抹除。剩下的只是一種美學風格，
一個可以被消費的符號——反抗變成了商品目錄裡的一個選項。這個過程是不可逆的：
一旦邊緣符號進入主流的符號經濟，它就無法再回到原來的位置。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "龐克主張技術匱乏是解放條件而非障礙，推倒音樂生產門檻讓任何人都可參與",
                "critique_target": ["專業門檻", "技術壟斷"],
                "value_axes": ["匱乏即解放", "去門檻化"],
                "materiality_axes": ["三個和弦", "簡陋器材"],
                "labor_time_axes": ["即時"],
                "abstract_patterns": ["匱乏→被迫創新→新方法", "限制→解放"],
                "theory_hints": ["DIY倫理", "無政府主義"]
            },
            {
                "core_claim": "DIY美學的粗糙是有意識的媒介誠實，材料的不完美本身是反機構的訊息",
                "critique_target": ["專業品質標準", "光滑表面"],
                "value_axes": ["媒介誠實", "粗糙即真實"],
                "materiality_axes": ["影印機", "手寫傳單", "卡帶"],
                "labor_time_axes": ["自行製作"],
                "abstract_patterns": ["媒介→忠於自身→正當性", "粗糙→真實性標記"],
                "theory_hints": ["DIY美學", "物質文化"]
            },
            {
                "core_claim": "龐克視覺語言被商業抽離生產脈絡後只剩可消費的符號，收編不可逆地消解了批判力量",
                "critique_target": ["商業收編", "符號消費化"],
                "value_axes": ["收編消解抵抗", "不可逆"],
                "materiality_axes": ["T-shirt", "做舊風格"],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→被收編→失去力量", "可見性→消解", "脈絡抽離→意義消失"],
                "theory_hints": ["Hebdige", "文化工業", "符號學"]
            }
        ],
        "thought_summary": "龐克DIY文化展現匱乏催生創新、媒介誠實、以及收編消解抵抗三重結構",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-7: 量子力學哲學（混雜：觀看即參與 + 邊緣定義 + 脈絡依賴）
# ──────────────────────────────────────────────────────────────────────
"kb_quantum": {
    "text": """
量子力學不只是一套物理理論，它從根本上挑戰了「觀察」這個概念的哲學預設。
在古典物理中，觀察者是透明的窗戶——你觀看世界，但你的觀看不改變世界。
量子力學打破了這個假設：測量一個電子的位置就會改變它的動量，觀察行為本身
就是系統的一部分。海森堡的不確定性原理不是技術限制（儀器不夠精確），
而是本體論的事實：粒子在被測量之前，不「擁有」確定的位置。

這帶出一個深刻的哲學問題：如果觀察改變了被觀察的對象，那麼「客觀現實」
意味著什麼？Niels Bohr 的哥本哈根詮釋走得更遠：他認為量子現象在被測量
之前根本不存在——不是我們不知道它的狀態，而是它沒有狀態。觀察不是發現，
而是創造。

從 Wittgenstein 的角度看，量子力學的「測量問題」也是一個關於邊界的問題。
什麼算是一次「測量」？一個蓋格計數器的咔嗒聲？一個實驗者看到的數據？
一隻貓在箱子裡的生死？（薛丁格的貓）這些邊界案例恰恰揭示了「觀察」這個
概念的模糊性——而正是這種模糊性，告訴我們「觀察」不是一個自然的範疇，
而是一個需要被哲學地審查的概念。

更值得注意的是量子力學中的「脈絡依賴性」(contextuality)。Bell 定理和
Kochen-Specker 定理表明，量子測量的結果不能被理解為物體「本來就有」的
屬性——它們依賴於測量的脈絡。同一個粒子，在不同的測量脈絡中給出不同的
結果。現實不是脈絡無關的，意義（測量值）由脈絡共同決定。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "量子力學打破觀察者透明性假設，觀察行為本身是系統的構成部分，不是被動接收",
                "critique_target": ["客觀旁觀者", "觀察者透明性"],
                "value_axes": ["觀察即參與", "主客不分"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["觀察→改變對象", "觀看→參與構成"],
                "theory_hints": ["海森堡", "哥本哈根詮釋", "Bohr"]
            },
            {
                "core_claim": "薛丁格的貓等邊界案例揭示了「觀察」概念本身的模糊性，邊緣案例定義了概念的本質",
                "critique_target": ["觀察作為自然範疇", "清晰邊界"],
                "value_axes": ["邊緣定義中心", "概念模糊性"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→定義中心", "模糊→揭示本質"],
                "theory_hints": ["Wittgenstein", "薛丁格的貓", "測量問題"]
            },
            {
                "core_claim": "量子測量結果依賴測量脈絡，現實不是脈絡無關的，意義由脈絡共同決定",
                "critique_target": ["脈絡無關的現實", "內在屬性論"],
                "value_axes": ["脈絡依賴", "脈絡即意義"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["脈絡→決定意義", "無脫脈絡的屬性"],
                "theory_hints": ["Bell定理", "Kochen-Specker", "脈絡主義"]
            }
        ],
        "thought_summary": "量子力學同時揭示觀察即參與、邊緣定義概念、脈絡決定意義三個哲學結構",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-8: 解構主義建築（混雜：過度→新秩序 + 約束→本質 + 媒介誠實）
# ──────────────────────────────────────────────────────────────────────
"kb_decon_arch": {
    "text": """
解構主義建築在 1980 年代的崛起常被誤解為純粹的形式遊戲——扭曲的牆面、
傾斜的柱子、看似即將倒塌的結構。但 Frank Gehry、Zaha Hadid、Daniel Libeskind
的實踐其實包含了深刻的結構邏輯。Gehry 的畢爾包古根漢美術館把鈦金屬板推到
了可加工性的極限——每一片曲面都是在材料即將失去結構強度的邊緣精確計算出來的。
這不是失控，而是在崩潰的臨界點上找到新的平衡。過度的變形，恰恰是新秩序湧現
的條件。

Libeskind 的柏林猶太博物館走的是另一個方向。銳角、斷裂的動線、故意的不舒適
感——這些不是審美選擇，而是試圖讓建築物忠實於它所紀念的歷史創傷。建築的
「形式不安」是對大屠殺記憶的材料回應。在這個意義上，解構建築的「誠實」不同於
清水模的「材料誠實」——它是一種「情感誠實」：形式不偽裝舒適，因為它紀念的
對象不舒適。

有趣的是，解構建築也展示了「約束產生本質」的另一種變體。傳統建築的約束是重力、
結構力學、使用功能；解構建築額外加入了「概念約束」——Libeskind 的設計被猶太
歷史的敘事結構所約束，每一個空間決定都必須回應這個敘事。這種雙重約束（物理+
概念）反而迫使設計達到了一種超越純形式的深度：空間不再只是空間，它同時是記憶、
是見證、是倫理立場的物質化。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "Gehry在材料失去結構強度的臨界點精確計算曲面，崩潰邊緣是新秩序湧現的條件",
                "critique_target": ["形式穩定性", "結構保守主義"],
                "value_axes": ["臨界", "新秩序湧現"],
                "materiality_axes": ["鈦金屬", "曲面"],
                "labor_time_axes": [],
                "abstract_patterns": ["過度→臨界→新秩序", "極端→崩潰→湧現"],
                "theory_hints": ["解構主義建築", "Gehry"]
            },
            {
                "core_claim": "Libeskind的建築形式不偽裝舒適因為紀念對象不舒適，這是情感層面的媒介誠實",
                "critique_target": ["舒適偽裝", "形式中立性"],
                "value_axes": ["情感誠實", "形式忠實"],
                "materiality_axes": ["銳角", "斷裂動線"],
                "labor_time_axes": [],
                "abstract_patterns": ["媒介→忠於自身→正當性", "形式→忠於情感內容"],
                "theory_hints": ["Libeskind", "記憶建築"]
            },
            {
                "core_claim": "解構建築的物理+概念雙重約束迫使空間超越純形式，成為記憶與倫理的物質化",
                "critique_target": ["純形式主義", "功能至上"],
                "value_axes": ["約束", "深度", "倫理物質化"],
                "materiality_axes": ["空間"],
                "labor_time_axes": [],
                "abstract_patterns": ["約束→剝除→本質浮現", "雙重限制→深度"],
                "theory_hints": ["記憶政治", "空間倫理"]
            }
        ],
        "thought_summary": "解構建築在臨界崩潰、情感誠實與約束深度三個維度展開",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-9: John Cage 與偶然音樂（混雜：留白/缺席 + 觀看即參與 + 邊緣定義）
# ──────────────────────────────────────────────────────────────────────
"kb_cage": {
    "text": """
John Cage 的《4'33"》在 1952 年首演時，鋼琴家 David Tudor 走上台、坐下、
打開琴蓋、沉默了四分三十三秒、然後合上琴蓋離開。觀眾憤怒了——他們覺得被愚弄。
但 Cage 的意圖恰恰相反：他要讓人聽見通常被音樂掩蓋的東西——觀眾的咳嗽聲、
椅子的嘎吱聲、外面的風聲。沉默不是沒有聲音，而是把「什麼算音樂」的判斷權
交還給聽眾。缺席是一種最主動的在場形式。

但《4'33"》更深的意義在於它如何改變了「聽」這個行為。在傳統音樂會中，
聽眾是被動的接收者——音樂從台上流向台下。但在《4'33"》中，聽眾突然發現
自己就是聲音的來源。你的咳嗽不是干擾，而是演出的一部分。聽的行為本身改變了
被聽到的內容——觀察改變了被觀察的對象。

同時，《4'33"》也是一個關於「音樂的邊界在哪裡」的實驗。如果沉默可以是音樂，
那麼音樂的定義就必須被重新審視。這跟 Duchamp 把小便斗放進美術館的邏輯完全
一樣：用一個最極端的邊緣案例來測試一個範疇的定義。Cage 和 Duchamp 都發現
了同一個結構：邊界比中心更能揭示一個概念的本質。

Cage 的偶然音樂（aleatory music）進一步把「缺席作為在場」的邏輯推廣。
他用擲幣來決定音符的序列——作曲家「不在場」的意志，反而讓音樂本身的可能性
空間完整地呈現。控制的缺席不是混亂，而是更大的秩序。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "Cage的沉默讓通常被掩蓋的聲音可見，缺席是最主動的在場形式",
                "critique_target": ["沉默=空無", "音樂需要聲音"],
                "value_axes": ["缺席即在場", "空即容器"],
                "materiality_axes": ["環境聲音", "沉默"],
                "labor_time_axes": ["四分三十三秒"],
                "abstract_patterns": ["缺席→主動在場", "空→揭示被掩蓋之物"],
                "theory_hints": ["John Cage", "4分33秒"]
            },
            {
                "core_claim": "《4'33\"》中聽眾從被動接收者變成聲音來源，聽的行為改變了被聽到的內容",
                "critique_target": ["被動聽眾", "單向傳遞"],
                "value_axes": ["觀看即參與", "聽者即演出者"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["觀察→改變對象", "接收者→參與者"],
                "theory_hints": ["接受美學", "互動藝術"]
            },
            {
                "core_claim": "沉默作為音樂的邊緣案例測試了音樂定義的極限，邊界比中心更能揭示本質",
                "critique_target": ["音樂需要聲音的預設", "範疇的固定邊界"],
                "value_axes": ["邊緣定義中心", "範疇測試"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→定義中心", "極端案例→揭示定義"],
                "theory_hints": ["Cage", "Duchamp", "概念藝術"]
            }
        ],
        "thought_summary": "Cage的實踐交織缺席即在場、觀看即參與、邊緣定義中心三個結構",
        "concepts": []
    }, ensure_ascii=False),
},

# ──────────────────────────────────────────────────────────────────────
# KB-10: 仕紳化與城市空間（混雜：收編消解 + 過度→新秩序 + 脈絡決定意義）
# ──────────────────────────────────────────────────────────────────────
"kb_gentrification": {
    "text": """
仕紳化的標準敘事是這樣的：藝術家搬進便宜的老社區，咖啡館和畫廊隨之出現，
媒體開始報導這個「有趣的」地方，開發商嗅到商機，租金上漲，最終藝術家和原住民
都被擠走。但這個過程的結構比表面看到的更複雜。

最核心的悖論是：創造者被自己創造的價值驅逐。藝術家為一個地方注入了「文化氛圍」
——但這種氛圍一旦被房地產市場識別為可量化的價值，它就啟動了一個自我消滅的循環。
邊緣的活力一旦被中心認可，就成為中心擴張的燃料。這跟龐克被商業收編的結構完全
一樣：可見性本身就是力量消解的開始。

但仕紳化也揭示了一個更微妙的現象：同一條街，在不同的時間點，承載著完全不同的
意義。十年前的小巷是「危險的、破敗的」；五年前變成「有個性的、真實的」；
現在則是「時尚的、可投資的」。街道本身沒有改變一塊磚，但它在不同的社會脈絡中
被賦予了完全不同的意義。脈絡不只決定了意義，它甚至決定了價值。

最有趣的是，仕紳化在最極端的階段會產生一種奇異的翻轉。當一個地區被完全
仕紳化——原住民全部遷走、獨立店鋪被連鎖品牌取代、所有「粗糙的邊緣」被
磨平——它就失去了最初吸引人的那種「真實性」。過度的開發反而摧毀了開發的基礎。
這跟解構建築把形式推到崩潰邊緣的邏輯有相似之處：過度的優化觸發了系統的自我
否定，從而可能（在最好的情況下）催生出對城市空間的全新理解。
""",
    "fake_resp": json.dumps({
        "claims": [
            {
                "core_claim": "仕紳化中創造者被自己創造的價值驅逐，可見性啟動力量消解的自毀循環",
                "critique_target": ["文化價值商品化", "開發邏輯"],
                "value_axes": ["收編消解抵抗", "自我消滅"],
                "materiality_axes": ["租金", "房地產"],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→被收編→失去力量", "成功→自毀"],
                "theory_hints": ["仕紳化理論", "David Harvey"]
            },
            {
                "core_claim": "同一條街在不同時間點因社會脈絡不同而被賦予完全不同的意義和價值",
                "critique_target": ["固定地方意義", "空間本質主義"],
                "value_axes": ["脈絡決定意義", "意義流動"],
                "materiality_axes": ["街道", "磚"],
                "labor_time_axes": ["時間推移"],
                "abstract_patterns": ["脈絡→決定意義", "改變脈絡→改變價值"],
                "theory_hints": ["Henri Lefebvre", "空間生產"]
            },
            {
                "core_claim": "完全仕紳化摧毀最初的吸引力，過度優化觸發系統自我否定，可能催生新理解",
                "critique_target": ["持續優化", "同質化"],
                "value_axes": ["過度觸發翻轉", "自我否定"],
                "materiality_axes": ["連鎖品牌", "磨平的邊緣"],
                "labor_time_axes": [],
                "abstract_patterns": ["過度→臨界→新秩序", "優化→自毀→可能重生"],
                "theory_hints": ["創造性破壞", "Schumpeter"]
            }
        ],
        "thought_summary": "仕紳化展現收編消解、脈絡決定意義、過度觸發翻轉三重結構",
        "concepts": []
    }, ensure_ascii=False),
},

}  # end KB_ARTICLES


# ═══════════════════════════════════════════════════════════════════════
# 10 個使用者對話場景 + 期望命中
# ═══════════════════════════════════════════════════════════════════════

# (name, user_turns, search_patterns, expected_kb_hits, expected_miss)
# expected_kb_hits: list of (doc_id, claim_index) — which specific claim should match
# expected_miss: doc_ids that should NOT be in the results

QUERY_SCENARIOS = [

    # Q1: 約束→本質 — 使用者談黑膠唱片的格式限制
    (
        "Q1: 約束→本質 (黑膠限制)",
        [
            {"role": "user", "content": "我最近在想黑膠唱片，每面只能放20分鐘的音樂"},
            {"role": "assistant", "content": "你覺得這個限制是壞事嗎？"},
            {"role": "user", "content": "反而不是。因為只有20分鐘，每首歌的存在都被放大了，音樂人不能塞進去水歌。限制讓每首歌都必須值得"},
        ],
        ["約束→剝除→本質浮現", "限制→放大感知"],
        {"kb_chado", "kb_concrete"},       # 茶道空間約束 + 住吉長屋
        {"kb_hiphop_sampling", "kb_gentrification"},  # 不該匹配
    ),

    # Q2: 時間倫理 — 使用者談慢食運動
    (
        "Q2: 時間倫理 (慢食運動)",
        [
            {"role": "user", "content": "慢食運動為什麼那麼堅持用傳統方法做東西"},
            {"role": "assistant", "content": "你認為他們堅持的是方法本身，還是方法背後的某個原則？"},
            {"role": "user", "content": "我覺得是時間。傳統方法慢不是因為沒有更快的替代方案，而是因為那個慢本身是品質的一部分。時間不是成本，時間是原料"},
        ],
        ["時間→材料化→意義", "不可壓縮→價值", "時間→凝固於成品"],
        {"kb_chado", "kb_concrete"},       # 茶道等待 + 清水模時間痕跡
        {"kb_duchamp", "kb_quantum"},      # 不該匹配
    ),

    # Q3: 媒介誠實 — 使用者談數位相機的HDR
    (
        "Q3: 媒介誠實 (HDR過度處理)",
        [
            {"role": "user", "content": "我越來越受不了手機拍照的HDR處理，每張照片都被修到不像真的"},
            {"role": "assistant", "content": "不像真的是指什麼？太亮？顏色太鮮豔？"},
            {"role": "user", "content": "是整個都在偽裝。明明是一個小感光元件，卻硬要做出大片的效果。我反而懷念底片相機那種顆粒感——它至少承認自己是什麼"},
        ],
        ["媒介→忠於自身→正當性", "拒絕偽裝→誠實"],
        {"kb_concrete", "kb_punk", "kb_chado", "kb_decon_arch"},  # 清水模誠實 + DIY誠實 + 茶碗 + Libeskind
        {"kb_hiphop_sampling", "kb_gentrification"},              # 不該匹配
    ),

    # Q4: 脈絡決定意義 — 使用者談同一首歌在不同場景
    (
        "Q4: 脈絡決定意義 (歌曲的場景)",
        [
            {"role": "user", "content": "同一首歌，在KTV唱是娛樂，在喪禮放是悼念，在廣告裡是行銷"},
            {"role": "assistant", "content": "你覺得歌本身有固定的意義嗎？"},
            {"role": "user", "content": "沒有。意義不在歌裡面，在它被放置的場景裡。抽掉場景，歌就只是聲波。脈絡不是包裝紙，脈絡就是意義本身"},
        ],
        ["脈絡→決定意義", "改變框架→改變意義", "無脫脈絡的意義"],
        {"kb_duchamp", "kb_translation", "kb_quantum", "kb_hiphop_sampling", "kb_gentrification"},
        {"kb_chado", "kb_concrete"},       # 不該匹配
    ),

    # Q5: 邊緣定義中心 — 使用者談非專業攝影
    (
        "Q5: 邊緣定義中心 (手機攝影)",
        [
            {"role": "user", "content": "現在人人都用手機拍照，你去問專業攝影師，他們覺得這不算攝影"},
            {"role": "assistant", "content": "你怎麼看？"},
            {"role": "user", "content": "我覺得恰好相反。正是那些「不算攝影」的手機照片——沒有構圖、沒有打光、只是按了快門——它們才最能說明「拍照」這件事到底是什麼。邊緣比中心更能定義一個東西"},
        ],
        ["邊緣→定義中心", "例外→揭示規則"],
        {"kb_duchamp", "kb_cage", "kb_quantum"},  # Duchamp邊緣 + Cage邊緣 + 量子邊緣
        {"kb_chado", "kb_concrete", "kb_gentrification"},
    ),

    # Q6: 過度→新秩序 — 使用者談電子音樂的噪音
    (
        "Q6: 過度→新秩序 (噪音音樂)",
        [
            {"role": "user", "content": "我去聽了一場噪音音樂的演出，完全不是我習慣的東西"},
            {"role": "assistant", "content": "聽起來怎麼樣？"},
            {"role": "user", "content": "一開始只覺得太吵太混亂。但聽了十分鐘之後，混亂裡開始出現某種韻律。不是音樂家安排的韻律，而是聲音碰撞自己產生的秩序。就好像把所有東西推到極端之後，另一種規則自己浮現了"},
        ],
        ["過度→臨界→新秩序", "極端→崩潰→湧現"],
        {"kb_decon_arch", "kb_gentrification"},  # Gehry臨界 + 仕紳化過度翻轉
        {"kb_chado", "kb_translation"},
    ),

    # Q7: 觀看即參與 — 使用者談逛展覽的體驗
    (
        "Q7: 觀看即參與 (策展觀察)",
        [
            {"role": "user", "content": "我帶一群不同背景的朋友去看同一個展，每個人看到的東西完全不一樣"},
            {"role": "assistant", "content": "有具體的例子嗎？"},
            {"role": "user", "content": "一幅抽象畫，工程師看到了力學張力，舞者看到了動態節奏，我看到的是色彩的情緒。不是我們解讀不同，是我們真的「看到」不同的東西。觀看本身在改變作品"},
        ],
        ["觀察→改變對象", "觀看→參與構成"],
        {"kb_quantum", "kb_cage", "kb_duchamp", "kb_translation"},  # 觀察改變 + Cage聽眾
        {"kb_punk", "kb_gentrification"},
    ),

    # Q8: 缺席即在場 — 使用者談建築中的院子
    (
        "Q8: 缺席即在場 (院子的空)",
        [
            {"role": "user", "content": "為什麼中國傳統建築的院子比房間更重要？"},
            {"role": "assistant", "content": "你說的「更重要」是什麼意思？"},
            {"role": "user", "content": "院子是空的，什麼都沒有建在上面。但那個「什麼都沒有」不是缺少——是刻意保留的。它讓光進來、讓風通過、讓人可以停下來。空不是虛，空是最重要的房間"},
        ],
        ["缺席→主動在場", "空→揭示被掩蓋之物"],
        {"kb_cage", "kb_chado"},           # Cage沉默 + 茶室留白（茶道文末提到空牆）
        {"kb_hiphop_sampling", "kb_punk"},
    ),

    # Q9: 收編消解抵抗 — 使用者談獨立書店
    (
        "Q9: 收編消解 (獨立書店)",
        [
            {"role": "user", "content": "我注意到越來越多大型連鎖書店開始做「獨立書店風格」的空間"},
            {"role": "assistant", "content": "你怎麼看這個趨勢？"},
            {"role": "user", "content": "很矛盾。獨立書店的美學——手寫標籤、不規則的書架排列、店主的個人選書——這些東西的力量來自它不是被設計出來的。當連鎖店複製這些外觀，保留了形式但抽掉了靈魂。邊緣被中心吸收的那一刻，就不再是邊緣了"},
        ],
        ["邊緣→被收編→失去力量", "可見性→消解"],
        {"kb_hiphop_sampling", "kb_punk", "kb_gentrification"},  # 嘻哈收編 + 龐克收編 + 仕紳化
        {"kb_quantum", "kb_translation"},
    ),

    # Q10: 轉換失敗→揭示 — 使用者談把小說改編成電影
    (
        "Q10: 轉換失敗→揭示 (小說改編)",
        [
            {"role": "user", "content": "每次看小說改編的電影都覺得「不是那個味道」"},
            {"role": "assistant", "content": "你覺得是電影做得不好，還是有些東西本來就轉換不了？"},
            {"role": "user", "content": "是後者。小說的內心獨白、時間的壓縮與膨脹、文字本身的節奏——這些東西在影像裡不存在。但有趣的是，正是這種「做不到」讓我更清楚小說獨特在哪裡。翻不過去的地方，恰好定義了原物的本質"},
        ],
        ["轉換失敗→揭示本質", "損失→認識"],
        {"kb_translation"},                # Benjamin翻譯論
        {"kb_quantum", "kb_gentrification"},
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════

async def run_tests():
    # ── Phase 1: Ingest all 10 KB articles into mega store ──
    print("\n" + "="*70)
    print("  Phase 1: Ingest 10 long-form KB articles into mega store")
    print("="*70)

    store = _store()
    all_claim_map: dict[str, list[Claim]] = {}  # doc_id → claims
    total_claims = 0

    for doc_id, info in KB_ARTICLES.items():
        text_len = len(info["text"].strip())
        extractor = ThoughtExtractor(
            llm_callable=_llm(info["fake_resp"]),
            project_id="mega",
        )
        result = await extractor.extract_from_passage(info["text"], source_id=doc_id)
        assert result.was_extracted, f"Extraction failed for {doc_id}"
        for claim in result.claims:
            await store.add(claim)
        all_claim_map[doc_id] = result.claims
        total_claims += len(result.claims)
        print(f"  ✓ {doc_id}: {len(result.claims)} claims ({text_len} chars)")
        for c in result.claims:
            patterns_str = ", ".join(c.abstract_patterns[:3])
            print(f"      [{patterns_str}] {c.core_claim[:45]}...")

    print(f"\n  Total: {total_claims} claims from {len(KB_ARTICLES)} articles")

    # ── Phase 2: Run 10 query scenarios ──
    print("\n" + "="*70)
    print("  Phase 2: Run 10 query scenarios against mega store")
    print("="*70)

    passed = 0
    failed = 0
    results_table: list[dict] = []

    for name, turns, patterns, expected_hits, expected_miss in QUERY_SCENARIOS:
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")
        user_msg = turns[-1]["content"]
        print(f"  User: \"{user_msg[:60]}...\"")
        print(f"  Patterns: {patterns}")

        # Structural search
        struct_results = await store.search_by_structure(
            abstract_patterns=patterns, limit=30,
        )
        found_docs = {r.source_id for r in struct_results}

        # Retriever route — pass patterns as concept_queries (simulating
        # what the pipeline would provide after query decomposition)
        retriever = Retriever(project_id="mega", claim_store=store)
        plan = RetrievalPlan(concept_queries=patterns)
        direct = await retriever._retrieve_from_claims(user_msg, plan=plan)
        direct_docs = {r["claim"].source_id for r in direct}

        all_found = found_docs | direct_docs

        # Calculate precision & recall
        true_pos = expected_hits & all_found
        false_neg = expected_hits - all_found
        false_pos = (expected_miss or set()) & all_found
        recall = len(true_pos) / len(expected_hits) if expected_hits else 1.0
        # precision relative to expected_miss (false positive rate)
        fp_rate = len(false_pos) / len(expected_miss) if expected_miss else 0.0

        ok = len(false_neg) == 0 and len(false_pos) == 0

        print(f"\n  Structural search found: {found_docs}")
        print(f"  Retriever found: {direct_docs}")
        print(f"  Expected hits: {expected_hits}")
        print(f"  Expected miss: {expected_miss}")
        print(f"  TP={len(true_pos)} FN={len(false_neg)} FP={len(false_pos)} "
              f"Recall={recall:.0%} FP_rate={fp_rate:.0%}")

        for r in struct_results:
            marker = "✓" if r.source_id in expected_hits else ("✗" if r.source_id in (expected_miss or set()) else "~")
            print(f"    {marker} [{r.source_id}] {r.core_claim[:50]}...")

        if ok:
            passed += 1
            print(f"  → PASS ✓")
        else:
            failed += 1
            if false_neg:
                print(f"  → FAIL: missing {false_neg}")
            if false_pos:
                print(f"  → FAIL: false positive {false_pos}")

        results_table.append({
            "name": name,
            "recall": recall,
            "fp_rate": fp_rate,
            "found": len(all_found),
            "ok": ok,
        })

    # ── Phase 3: Summary ──
    print("\n" + "="*70)
    print("  Phase 3: Summary")
    print("="*70)

    avg_recall = sum(r["recall"] for r in results_table) / len(results_table)
    avg_fp = sum(r["fp_rate"] for r in results_table) / len(results_table)

    print(f"\n  {'Scenario':<45} {'Recall':>8} {'FP Rate':>8} {'Result':>8}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")
    for r in results_table:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  {r['name']:<45} {r['recall']:>7.0%} {r['fp_rate']:>7.0%} {status:>8}")
    print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'AVERAGE':<45} {avg_recall:>7.0%} {avg_fp:>7.0%}")
    print(f"\n  Passed: {passed}/{len(QUERY_SCENARIOS)}, Failed: {failed}/{len(QUERY_SCENARIOS)}")

    if failed == 0:
        print(f"\n  ALL TESTS PASSED ✓")
    else:
        print(f"\n  ⚠ {failed} TESTS FAILED")

    return failed == 0


if __name__ == "__main__":
    print("="*70)
    print("  De-insight: Deep Mode Long-Form Mixed-Concept Tests")
    print("  10 articles × 500+ chars × 2-3 claims each × 10 query scenarios")
    print("="*70)
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
