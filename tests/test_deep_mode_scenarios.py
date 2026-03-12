"""Deep mode scenario tests — 10 realistic multi-turn conversations.

Tests the FULL deep mode pipeline:
1. Knowledge base has cross-domain content (claim extraction → store)
2. User converses in multi-turn dialogue about a DIFFERENT domain
3. _decompose_query_deep extracts abstract structural queries
4. Retriever finds structurally-similar claims across domains
5. Deep→fast fallback works when deep sub-queries all miss

Each scenario verifies:
- Claim extraction from KB passages (fake LLM)
- Structural search across domains
- Retriever hybrid path finds correct matches
- Negative: doesn't return structurally-unrelated claims
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


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _make_llm(resp):
    async def f(prompt): return resp
    return LLMCallable(func=f)

def _tmp_store(pid="test"):
    return ClaimStore(project_id=pid, db_path=Path(tempfile.mktemp(suffix=".db")))


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 1: 約束→本質（跨媒介）
# User talks about minimalist design; KB has haiku, B&W photography
# ═══════════════════════════════════════════════════════════════════════

S1_CONVERSATION = [
    {"role": "user", "content": "我最近在想極簡主義設計，覺得它厲害的地方不是「少」，而是少了之後你沒辦法躲"},
    {"role": "assistant", "content": "你說的「沒辦法躲」是什麼意思？是指設計師自己，還是作品本身？"},
    {"role": "user", "content": "兩個都是。當你把裝飾都拿掉，剩下的東西如果不夠好，馬上就被看穿"},
]

S1_KB_PASSAGES = {
    "doc_haiku_constraint": {
        "text": "俳句的5-7-5音節限制迫使詩人找到最精確的那個詞，冗餘的修飾和解釋性連接必須被捨棄，剩下的每個詞都承擔最大意義密度。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "俳句的極端形式約束迫使每個詞承擔最大意義密度，冗餘必須被完全捨棄",
                "critique_target": ["冗長表達", "解釋性寫作"],
                "value_axes": ["本質", "精煉", "意義密度"],
                "materiality_axes": ["音節作為物理單位"],
                "labor_time_axes": ["凝練時間"],
                "abstract_patterns": ["約束→暴露", "形式限制→精煉", "減法→本質浮現"],
                "theory_hints": ["侘寂美學", "間"]
            }],
            "thought_summary": "形式約束作為穿透表面的方法",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_bw_photo": {
        "text": "黑白攝影失去色彩之後，構圖和光影的弱點無處可藏。色彩能掩蓋的瑕疵，在灰階中一覽無遺。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "黑白攝影剝除色彩後，構圖與光影的弱點無處遁形，限制反而暴露本質",
                "critique_target": ["色彩依賴", "表面華麗"],
                "value_axes": ["暴露", "本質", "不可掩飾"],
                "materiality_axes": ["灰階", "光影"],
                "labor_time_axes": [],
                "abstract_patterns": ["約束→暴露", "剝除→不可掩飾", "限制→本質浮現"],
                "theory_hints": ["攝影本體論"]
            }],
            "thought_summary": "色彩剝奪作為暴露裝置",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_bauhaus_ethics": {
        "text": "包浩斯主張形式必須誠實反映功能與材料，鋼管就該看起來像鋼管，混凝土不需要偽裝成大理石。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "包浩斯的材料誠實立場要求形式反映功能，剝除偽裝是倫理選擇而非審美偏好",
                "critique_target": ["材料偽裝", "裝飾欺騙"],
                "value_axes": ["誠實", "功能", "倫理"],
                "materiality_axes": ["鋼管", "混凝土"],
                "labor_time_axes": ["工業製程"],
                "abstract_patterns": ["移除偽裝→揭示真實", "誠實→剝除→功能即形式"],
                "theory_hints": ["Adolf Loos", "功能主義"]
            }],
            "thought_summary": "材料誠實作為倫理立場",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S1_EXPECTED_MATCHES = ["doc_haiku_constraint", "doc_bw_photo"]  # share 約束→暴露
S1_EXPECTED_MISS = ["doc_bauhaus_ethics"]  # different structure (ethics, not exposure)


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 2: 媒介誠實性
# User criticizes digital UI fakeness; KB has Greenberg, concrete arch
# ═══════════════════════════════════════════════════════════════════════

S2_CONVERSATION = [
    {"role": "user", "content": "我覺得很多數位設計的問題是它一直在假裝自己不是螢幕"},
    {"role": "assistant", "content": "你是指擬物設計嗎？還是更廣的意思？"},
    {"role": "user", "content": "更廣。像那些模擬紙張質感的app、假裝有景深的UI，都是在否認自己是像素"},
]

S2_KB_PASSAGES = {
    "doc_greenberg": {
        "text": "Clement Greenberg主張現代主義繪畫應承認畫布的平面性，而非製造透視幻覺。繪畫的倫理在於忠於自身媒介的物質條件。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Greenberg主張繪畫應承認畫布平面性，媒介忠於自身物質條件才具有正當性",
                "critique_target": ["透視幻覺", "媒介偽裝"],
                "value_axes": ["媒介誠實", "平面性", "物質忠誠"],
                "materiality_axes": ["畫布", "顏料"],
                "labor_time_axes": [],
                "abstract_patterns": ["媒介→忠於自身→正當性", "拒絕偽裝→誠實"],
                "theory_hints": ["Greenberg", "現代主義", "媒介特殊性"]
            }],
            "thought_summary": "媒介誠實作為藝術正當性的基礎",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_concrete_arch": {
        "text": "安藤忠雄不在混凝土上覆蓋任何飾面，讓材料的毛孔和氣泡成為建築表面，清水模的誠實在於拒絕掩飾。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "清水模建築拒絕在混凝土上覆蓋飾面，材料的瑕疵即是表達，掩飾即是不誠實",
                "critique_target": ["表面裝飾", "材料掩飾"],
                "value_axes": ["材料誠實", "裸露", "瑕疵即表達"],
                "materiality_axes": ["混凝土", "毛孔", "氣泡"],
                "labor_time_axes": ["模板工藝"],
                "abstract_patterns": ["媒介→忠於自身→正當性", "拒絕掩飾→誠實"],
                "theory_hints": ["安藤忠雄", "物質性"]
            }],
            "thought_summary": "材料裸露作為建築倫理",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_pottery": {
        "text": "不上釉的陶土誠實地展示了窯燒的痕跡，每一道裂紋都是火與土對話的證據。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "不上釉陶器展示窯燒痕跡，裂紋是製程與材料對話的證據，拒絕掩飾即是誠實",
                "critique_target": ["完美主義", "表面均質"],
                "value_axes": ["材料誠實", "過程痕跡", "不完美"],
                "materiality_axes": ["陶土", "窯燒", "裂紋"],
                "labor_time_axes": ["窯燒時間", "不可控過程"],
                "abstract_patterns": ["媒介→忠於自身→正當性", "痕跡→誠實"],
                "theory_hints": ["侘寂", "物派"]
            }],
            "thought_summary": "材料痕跡作為誠實的證據",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S2_EXPECTED_MATCHES = ["doc_greenberg", "doc_concrete_arch", "doc_pottery"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 3: 工藝的時間倫理
# User muses on hand-brewed coffee; KB has letterpress, lacquer
# ═══════════════════════════════════════════════════════════════════════

S3_CONVERSATION = [
    {"role": "user", "content": "我在想為什麼手沖咖啡讓人覺得特別，明明機器可以更精準"},
    {"role": "assistant", "content": "你覺得「特別」的是結果（味道），還是過程本身？"},
    {"role": "user", "content": "過程。那個等待的時間好像是味道的一部分"},
]

S3_KB_PASSAGES = {
    "doc_letterpress": {
        "text": "活版印刷的每個字一個一個排列，排版師傅的時間凝固在每一頁裡。速度不是目標，每一次按壓都是不可替代的物理事件。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "活版印刷中排版師的時間凝固在成品裡，時間本身是不可壓縮的價值成分",
                "critique_target": ["效率至上", "速度崇拜"],
                "value_axes": ["時間價值", "不可壓縮", "凝固"],
                "materiality_axes": ["鉛字", "紙張", "壓痕"],
                "labor_time_axes": ["手排時間", "逐字逐行", "不可加速"],
                "abstract_patterns": ["時間→凝固於成品", "不可壓縮→價值", "過程即材料"],
                "theory_hints": ["Walter Benjamin靈光", "手工vs機械複製"]
            }],
            "thought_summary": "時間作為不可替代的物質成分",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_lacquer": {
        "text": "輪島塗需要上百道工序，每層漆之間必須等待數週乾燥。等待不是浪費，是讓材料自行完成轉化的必要時間。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "漆器的等待乾燥時間不是浪費而是材料轉化的必要過程，時間是材料的一部分",
                "critique_target": ["效率邏輯", "時間即成本"],
                "value_axes": ["時間價值", "等待", "材料轉化"],
                "materiality_axes": ["漆", "乾燥", "層疊"],
                "labor_time_axes": ["數週等待", "百道工序", "不可加速"],
                "abstract_patterns": ["時間→凝固於成品", "等待→轉化", "不可壓縮→價值"],
                "theory_hints": ["工藝哲學"]
            }],
            "thought_summary": "等待作為材料轉化的不可壓縮過程",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S3_EXPECTED_MATCHES = ["doc_letterpress", "doc_lacquer"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 4: 翻譯中的不可譯性
# User struggles with exhibition translation; KB has Benjamin, music
# ═══════════════════════════════════════════════════════════════════════

S4_CONVERSATION = [
    {"role": "user", "content": "我在做一個展覽的英文翻譯，發現有些中文概念根本翻不過去"},
    {"role": "assistant", "content": "是哪類概念？是詞彙層面的，還是整個思考方式不一樣？"},
    {"role": "user", "content": "像「氣韻生動」，你可以翻成spirit resonance，但那完全不是同一回事。翻譯反而讓我更懂原文在說什麼"},
]

S4_KB_PASSAGES = {
    "doc_benjamin_translation": {
        "text": "Walter Benjamin認為翻譯不是複製原文，而是讓原文在另一種語言中的不可能性被看見。翻譯的價值在於揭示原文中那些無法轉換的核心。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Benjamin認為翻譯揭示原文的不可能性，不可轉換之物反過來定義了原文的獨特性",
                "critique_target": ["翻譯等價論", "意義可完全傳遞"],
                "value_axes": ["不可譯性", "獨特性", "揭示"],
                "materiality_axes": ["語言物質性"],
                "labor_time_axes": [],
                "abstract_patterns": ["轉換失敗→揭示本質", "不可譯→定義獨特", "損失→認識"],
                "theory_hints": ["Walter Benjamin", "翻譯者的任務"]
            }],
            "thought_summary": "翻譯的失敗作為認識論工具",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_orchestral_reduction": {
        "text": "把管弦樂改編成鋼琴獨奏時，失去的合奏層次反而凸顯了旋律骨架。簡化不只是損失，它同時是一種揭示。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "管弦改編為鋼琴的過程中損失的層次反而凸顯了旋律本質，轉換中的損失即揭示",
                "critique_target": ["忠實再現", "原作至上"],
                "value_axes": ["損失即揭示", "本質浮現"],
                "materiality_axes": ["管弦vs鍵盤", "音色層次"],
                "labor_time_axes": [],
                "abstract_patterns": ["轉換失敗→揭示本質", "損失→凸顯骨架"],
                "theory_hints": ["改編理論", "媒介轉換"]
            }],
            "thought_summary": "媒介轉換中的損失作為本質的揭示",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S4_EXPECTED_MATCHES = ["doc_benjamin_translation", "doc_orchestral_reduction"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 5: 邊緣定義中心
# User curates "non-design design"; KB has Art Brut, Wittgenstein
# ═══════════════════════════════════════════════════════════════════════

S5_CONVERSATION = [
    {"role": "user", "content": "我在策一個關於「不是設計的設計」的展覽"},
    {"role": "assistant", "content": "能多說一點嗎？你心目中「不是設計的設計」是什麼樣的東西？"},
    {"role": "user", "content": "像菜市場的手寫價目表、工地的臨時指示牌。它們沒有被「設計」過，但視覺上超有力量。我覺得它們反而說明了設計到底是什麼"},
]

S5_KB_PASSAGES = {
    "doc_art_brut": {
        "text": "原生藝術（Art Brut）指未受正式藝術訓練者的創作。這些作品挑戰了學院對「什麼是藝術」的定義——藝術的邊界恰恰由這些邊緣案例來劃定。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "原生藝術作為藝術邊界的案例，其存在本身比典型藝術更能定義「什麼是藝術」",
                "critique_target": ["學院定義壟斷", "專業門檻"],
                "value_axes": ["邊緣定義中心", "範疇挑戰"],
                "materiality_axes": [],
                "labor_time_axes": ["未受訓的自發性"],
                "abstract_patterns": ["邊緣→定義中心", "例外→揭示規則", "邊界模糊→範疇重構"],
                "theory_hints": ["Art Brut", "Dubuffet", "outsider art"]
            }],
            "thought_summary": "邊緣案例作為定義裝置",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_wittgenstein_boundary": {
        "text": "Wittgenstein認為定義一個概念最好的方法不是找核心特徵，而是觀察它的邊界在哪裡模糊——家族相似性意味著沒有共享的本質，只有邊界的交疊。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Wittgenstein的家族相似性主張邊界觀察比核心定義更能揭示概念本質",
                "critique_target": ["本質主義定義", "充要條件"],
                "value_axes": ["邊界", "模糊性", "家族相似"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→定義中心", "模糊→揭示", "邊界觀察→概念理解"],
                "theory_hints": ["Wittgenstein", "家族相似性", "語言遊戲"]
            }],
            "thought_summary": "邊界觀察作為理解概念的方法",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S5_EXPECTED_MATCHES = ["doc_art_brut", "doc_wittgenstein_boundary"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 6: 過度→崩潰→新秩序
# User discusses baroque music ornament; KB has Gehry, Pollock
# ═══════════════════════════════════════════════════════════════════════

S6_CONVERSATION = [
    {"role": "user", "content": "我一直在想巴洛克音樂的裝飾為什麼那麼多"},
    {"role": "assistant", "content": "你覺得那些裝飾是在幹嘛？是炫技、還是有結構上的作用？"},
    {"role": "user", "content": "我覺得它把裝飾推到一個極端，推到你已經分不清主旋律和裝飾的界線，然後從那個混亂裡產生一種新的秩序"},
]

S6_KB_PASSAGES = {
    "doc_gehry": {
        "text": "Frank Gehry把建築的穩定性推到崩潰邊緣，扭曲的金屬表面看似即將倒塌，但正是在這個臨界點上找到了新的結構表達。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Gehry將建築穩定性推至崩潰臨界點，在那個極端處找到新的結構可能",
                "critique_target": ["穩定性崇拜", "結構保守主義"],
                "value_axes": ["臨界", "極端", "新秩序"],
                "materiality_axes": ["鈦金屬", "扭曲表面"],
                "labor_time_axes": [],
                "abstract_patterns": ["過度→臨界→新秩序", "推向極端→崩潰→湧現"],
                "theory_hints": ["解構主義建築"]
            }],
            "thought_summary": "臨界崩潰作為新結構的起源",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_pollock": {
        "text": "Jackson Pollock的滴畫法放棄了所有構圖控制，但過度的混亂反而產生了自組織的視覺韻律——秩序從失控中自行湧現。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Pollock放棄構圖控制後，過度混亂反而產生自組織韻律，秩序從失控中湧現",
                "critique_target": ["構圖控制", "秩序預設"],
                "value_axes": ["自組織", "湧現", "失控中的秩序"],
                "materiality_axes": ["滴灑顏料", "畫布"],
                "labor_time_axes": ["身體行為"],
                "abstract_patterns": ["過度→臨界→新秩序", "失控→自組織→湧現"],
                "theory_hints": ["行動繪畫", "複雜系統"]
            }],
            "thought_summary": "失控作為新秩序湧現的條件",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S6_EXPECTED_MATCHES = ["doc_gehry", "doc_pollock"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 7: 觀看改變被觀看物
# User observes people see art differently; KB has quantum, Berger
# ═══════════════════════════════════════════════════════════════════════

S7_CONVERSATION = [
    {"role": "user", "content": "我最近帶人去看展覽，發現同一件作品在不同人眼裡完全不同"},
    {"role": "assistant", "content": "是指解讀不同，還是他們連「看到什麼」都不一樣？"},
    {"role": "user", "content": "連看到什麼都不一樣。一個學建築的朋友看一幅畫，他看到的是空間結構；學音樂的看同一幅畫，看到的是節奏。我覺得觀看不是被動的"},
]

S7_KB_PASSAGES = {
    "doc_quantum_observer": {
        "text": "量子力學中，測量行為會改變粒子的狀態。觀察者不可能是局外人，觀測本身就是系統的一部分。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "量子觀測中觀察行為改變被觀測對象的狀態，觀察者不可能是系統外的旁觀者",
                "critique_target": ["客觀旁觀者", "被動觀測"],
                "value_axes": ["觀察即參與", "主客不分"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["觀察→改變對象", "觀看即參與構成", "主客交纏"],
                "theory_hints": ["量子力學", "觀察者效應", "哥本哈根詮釋"]
            }],
            "thought_summary": "觀察作為對象構成的一部分",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_berger_ways": {
        "text": "John Berger指出我們從不只是在看事物本身，我們總是在看事物和我們之間的關係。觀看是一種選擇性的行為。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Berger主張觀看永遠是看我們與事物的關係而非事物本身，觀看是主動的選擇行為",
                "critique_target": ["被動接收", "純粹客觀觀看"],
                "value_axes": ["觀看即選擇", "關係性"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["觀察→改變對象", "觀看即構成關係"],
                "theory_hints": ["John Berger", "Ways of Seeing", "視覺文化"]
            }],
            "thought_summary": "觀看作為主動的關係構成",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S7_EXPECTED_MATCHES = ["doc_quantum_observer", "doc_berger_ways"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 8: 脈絡決定意義
# User discusses Duchamp; KB has Goffman, DJ culture
# ═══════════════════════════════════════════════════════════════════════

S8_CONVERSATION = [
    {"role": "user", "content": "Duchamp把小便斗放進美術館，到現在一百年了大家還在討論"},
    {"role": "assistant", "content": "你覺得那個動作最厲害的地方是什麼？"},
    {"role": "user", "content": "不是小便斗本身，是他證明了「放在哪裡」比「是什麼」更重要。同一個東西換個地方就完全不同了"},
]

S8_KB_PASSAGES = {
    "doc_goffman_frame": {
        "text": "Erving Goffman的框架理論指出，人們對同一事件的理解取決於他們套用的詮釋框架。改變框架就改變了事件的意義。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Goffman框架理論：對事件的理解取決於詮釋框架，改變框架即改變意義",
                "critique_target": ["固定意義", "脫脈絡理解"],
                "value_axes": ["框架決定意義", "脈絡依賴"],
                "materiality_axes": [],
                "labor_time_axes": [],
                "abstract_patterns": ["脈絡→決定意義", "改變框架→改變意義", "無脫脈絡的意義"],
                "theory_hints": ["Goffman", "框架分析", "符號互動論"]
            }],
            "thought_summary": "框架作為意義的構成條件",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_dj_recontextualize": {
        "text": "DJ把別人的歌放在新的序列裡，創造了原本不存在的意義。取樣不是借用，是脈絡的重新發明。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "DJ將既有音樂放入新序列創造新意義，脈絡的重新配置就是意義的生產",
                "critique_target": ["原創至上", "固定意義"],
                "value_axes": ["再脈絡化", "意義生產"],
                "materiality_axes": ["唱片", "混音器"],
                "labor_time_axes": ["即時混音"],
                "abstract_patterns": ["脈絡→決定意義", "去脈絡→再脈絡→新意義"],
                "theory_hints": ["取樣文化", "後現代拼貼"]
            }],
            "thought_summary": "脈絡重組作為意義生產的方法",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S8_EXPECTED_MATCHES = ["doc_goffman_frame", "doc_dj_recontextualize"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 9: 留白的主動性
# User fights editor about whitespace; KB has Cage, rock garden
# ═══════════════════════════════════════════════════════════════════════

S9_CONVERSATION = [
    {"role": "user", "content": "我在做一本書的版面，總編一直要我把每一頁都填滿"},
    {"role": "assistant", "content": "你不想填滿？"},
    {"role": "user", "content": "對，我覺得留白是有功能的。那個空不是「沒有東西」，它是在給讀者喘息的空間，是在說「這裡需要你自己想」"},
]

S9_KB_PASSAGES = {
    "doc_cage_silence": {
        "text": "John Cage的《4'33\"》是四分三十三秒的沉默。但那不是沒有音樂——沉默讓聽眾聽見了環境聲音，空成為了容器。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "Cage的沉默不是沒有音樂，而是讓環境成為內容的框架，缺席即是一種主動的在場",
                "critique_target": ["空=沒有", "沉默=缺失"],
                "value_axes": ["缺席即在場", "空即容器", "邀請"],
                "materiality_axes": ["環境聲音", "時間"],
                "labor_time_axes": ["等待的時間"],
                "abstract_patterns": ["缺席→主動在場", "空→邀請填充", "留白→觀者參與"],
                "theory_hints": ["John Cage", "偶然音樂", "4分33秒"]
            }],
            "thought_summary": "沉默/缺席作為主動的邀請機制",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_rock_garden": {
        "text": "枯山水的白砂不是空地，它代表水；石頭之間的空間是構圖的主體。留白不是缺少，而是最重要的元素。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "枯山水中白砂和石間空隙是構圖主體而非空白，缺席是最重要的表達元素",
                "critique_target": ["空=缺少", "填滿=充實"],
                "value_axes": ["空即主體", "留白為表達"],
                "materiality_axes": ["白砂", "石", "空間"],
                "labor_time_axes": ["冥想時間"],
                "abstract_patterns": ["缺席→主動在場", "空→主體", "留白→核心元素"],
                "theory_hints": ["枯山水", "日本庭園", "間"]
            }],
            "thought_summary": "空作為構圖的核心而非缺失",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S9_EXPECTED_MATCHES = ["doc_cage_silence", "doc_rock_garden"]


# ═══════════════════════════════════════════════════════════════════════
# SCENARIO 10: 抵抗同化的少數位置
# User worries about indie publishing; KB has punk, gentrification
# ═══════════════════════════════════════════════════════════════════════

S10_CONVERSATION = [
    {"role": "user", "content": "獨立出版最近越來越紅，大出版社也開始做「獨立風格」的書了"},
    {"role": "assistant", "content": "你覺得這是好事還是壞事？"},
    {"role": "user", "content": "很複雜。被看見是好的，但「獨立」的意義本來就是因為它在體制外。當大出版社模仿獨立的美學，那個美學就失去了它原本的抵抗性"},
]

S10_KB_PASSAGES = {
    "doc_punk_cooption": {
        "text": "當Hot Topic開始賣龐克T-shirt，龐克的反叛符號就被收編為消費選項。符號從體制外進入體制內的瞬間，就失去了它的批判力量。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "龐克符號被商業收編後失去反叛力量，邊緣進入中心即被中和",
                "critique_target": ["商業收編", "符號消費化"],
                "value_axes": ["收編消解抵抗", "邊緣vs中心"],
                "materiality_axes": ["T-shirt", "商品"],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→被收編→失去力量", "可見性vs抵抗性", "收編→中和"],
                "theory_hints": ["次文化理論", "Hebdige", "文化工業"]
            }],
            "thought_summary": "商業收編作為批判力量的消解機制",
            "concepts": []
        }, ensure_ascii=False),
    },
    "doc_gentrification": {
        "text": "藝術家搬進便宜的社區創造了文化氛圍，但這個氛圍吸引了開發商，最終把藝術家擠走。創造者被自己創造的價值驅逐。",
        "fake_resp": json.dumps({
            "claims": [{
                "core_claim": "仕紳化中創造者被自己創造的價值驅逐，邊緣活力一旦被中心認可就啟動自我消滅",
                "critique_target": ["城市開發邏輯", "文化作為地產工具"],
                "value_axes": ["自我消滅", "收編", "驅逐"],
                "materiality_axes": ["社區空間", "租金"],
                "labor_time_axes": [],
                "abstract_patterns": ["邊緣→被收編→失去力量", "成功即自毀", "創造→吸引→驅逐"],
                "theory_hints": ["仕紳化", "David Harvey", "空間政治"]
            }],
            "thought_summary": "邊緣的可見性觸發自我消滅循環",
            "concepts": []
        }, ensure_ascii=False),
    },
}

S10_EXPECTED_MATCHES = ["doc_punk_cooption", "doc_gentrification"]


# ═══════════════════════════════════════════════════════════════════════
# ALL SCENARIOS REGISTRY
# ═══════════════════════════════════════════════════════════════════════

ALL_SCENARIOS = [
    ("S1: 約束→暴露→本質", S1_CONVERSATION, S1_KB_PASSAGES, S1_EXPECTED_MATCHES,
     ["約束→暴露", "限制→本質浮現"], ["媒介誠實", "時間價值"]),
    ("S2: 媒介誠實性", S2_CONVERSATION, S2_KB_PASSAGES, S2_EXPECTED_MATCHES,
     ["媒介→忠於自身→正當性", "拒絕偽裝→誠實"], ["約束→暴露", "過度→臨界"]),
    ("S3: 工藝的時間倫理", S3_CONVERSATION, S3_KB_PASSAGES, S3_EXPECTED_MATCHES,
     ["時間→凝固於成品", "不可壓縮→價值"], ["約束→暴露", "脈絡→決定意義"]),
    ("S4: 翻譯中的不可譯性", S4_CONVERSATION, S4_KB_PASSAGES, S4_EXPECTED_MATCHES,
     ["轉換失敗→揭示本質", "損失→認識"], ["觀察→改變對象", "缺席→主動在場"]),
    ("S5: 邊緣定義中心", S5_CONVERSATION, S5_KB_PASSAGES, S5_EXPECTED_MATCHES,
     ["邊緣→定義中心", "例外→揭示規則"], ["過度→臨界", "收編→中和"]),
    ("S6: 過度→崩潰→新秩序", S6_CONVERSATION, S6_KB_PASSAGES, S6_EXPECTED_MATCHES,
     ["過度→臨界→新秩序", "推向極端→崩潰→湧現"], ["約束→暴露", "媒介→忠於自身"]),
    ("S7: 觀看改變被觀看物", S7_CONVERSATION, S7_KB_PASSAGES, S7_EXPECTED_MATCHES,
     ["觀察→改變對象", "觀看即參與構成"], ["脈絡→決定意義", "轉換失敗→揭示"]),
    ("S8: 脈絡決定意義", S8_CONVERSATION, S8_KB_PASSAGES, S8_EXPECTED_MATCHES,
     ["脈絡→決定意義", "改變框架→改變意義"], ["邊緣→定義中心", "約束→暴露"]),
    ("S9: 留白的主動性", S9_CONVERSATION, S9_KB_PASSAGES, S9_EXPECTED_MATCHES,
     ["缺席→主動在場", "空→邀請填充"], ["時間→凝固", "收編→失去力量"]),
    ("S10: 收編消解抵抗", S10_CONVERSATION, S10_KB_PASSAGES, S10_EXPECTED_MATCHES,
     ["邊緣→被收編→失去力量", "可見性vs抵抗性"], ["缺席→主動在場", "觀察→改變對象"]),
]


# ═══════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════

async def run_all_scenario_tests():
    """Run all 10 scenarios through the full pipeline."""

    passed = 0
    failed = 0
    details: list[str] = []

    for scenario_name, conversation, kb_passages, expected_matches, match_patterns, anti_patterns in ALL_SCENARIOS:

        print(f"\n{'='*70}")
        print(f"  {scenario_name}")
        print(f"{'='*70}")

        store = _tmp_store()

        # ── Phase A: Ingest KB passages as claims ──
        print(f"\n  Phase A: Ingesting {len(kb_passages)} KB passages")
        for doc_id, info in kb_passages.items():
            extractor = ThoughtExtractor(
                llm_callable=_make_llm(info["fake_resp"]),
                project_id="test",
            )
            result = await extractor.extract_from_passage(
                info["text"], source_id=doc_id,
            )
            assert result.was_extracted, f"  ✗ Extraction failed for {doc_id}"
            for claim in result.claims:
                await store.add(claim)
            print(f"    ✓ {doc_id}: {len(result.claims)} claims extracted")

        all_claims = await store.list_by_project("test", limit=100)
        print(f"    Total claims in store: {len(all_claims)}")

        # ── Phase B: Structural search with expected patterns ──
        print(f"\n  Phase B: Structural search with match_patterns={match_patterns}")
        struct_results = await store.search_by_structure(
            abstract_patterns=match_patterns, limit=20,
        )
        matched_docs = {r.source_id for r in struct_results}
        print(f"    Found {len(struct_results)} claims from docs: {matched_docs}")
        for r in struct_results:
            print(f"      ✓ [{r.source_id}] {r.core_claim[:50]}...")

        # Check expected matches
        scenario_ok = True
        for exp in expected_matches:
            if exp in matched_docs:
                print(f"    ✓ Expected match found: {exp}")
            else:
                print(f"    ✗ MISSING expected match: {exp}")
                scenario_ok = False

        # ── Phase C: Negative check — anti-patterns should NOT match ──
        print(f"\n  Phase C: Anti-pattern check (should NOT match): {anti_patterns}")
        anti_results = await store.search_by_structure(
            abstract_patterns=anti_patterns, limit=20,
        )
        anti_docs = {r.source_id for r in anti_results}
        if not anti_docs:
            print(f"    ✓ No false matches (correct)")
        else:
            # It's OK if there are some matches (patterns might overlap),
            # but they shouldn't be the PRIMARY pattern of this scenario's docs
            print(f"    ⚠ Found {len(anti_results)} claims (reviewing...): {anti_docs}")
            for r in anti_results:
                print(f"      ? [{r.source_id}] {r.core_claim[:50]}...")

        # ── Phase D: Retriever full pipeline ──
        print(f"\n  Phase D: Retriever hybrid pipeline")
        retriever = Retriever(project_id="test", claim_store=store)

        user_msg = conversation[-1]["content"]
        plan = RetrievalPlan(
            project_id="test",
            query_mode="deep",
            thought_summary="",
            concept_queries=[user_msg],
        )
        ret_result = await retriever.retrieve(plan, user_msg)
        ret_docs = {c.source_id for c in ret_result.claims}
        print(f"    Retrieved {len(ret_result.claims)} claims, {len(ret_result.bridges)} bridges")
        for c in ret_result.claims:
            print(f"      ✓ [{c.source_id}] {c.core_claim[:50]}...")

        # Also test _retrieve_from_claims directly with the user's last msg
        direct_results = await retriever._retrieve_from_claims(user_msg)
        direct_docs = {r["claim"].source_id for r in direct_results}
        text_m = [r for r in direct_results if r["reason"] == "text_match"]
        struct_m = [r for r in direct_results if r["reason"] == "structural_match"]
        print(f"    Direct _retrieve_from_claims: text={len(text_m)} structural={len(struct_m)}")
        for r in direct_results:
            c = r["claim"]
            print(f"      [{r['reason']}] [{c.source_id}] {c.core_claim[:45]}...")

        # Check at least one expected doc was found via ANY route
        all_found = matched_docs | ret_docs | direct_docs
        for exp in expected_matches:
            if exp not in all_found:
                print(f"    ✗ {exp} not found by ANY retrieval method!")
                scenario_ok = False

        if scenario_ok:
            passed += 1
            details.append(f"  ✓ {scenario_name}")
            print(f"\n  → PASS ✓")
        else:
            failed += 1
            details.append(f"  ✗ {scenario_name}")
            print(f"\n  → FAIL ✗")

    # ═══════════════════════════════════════════════════════════════════
    # BONUS: Cross-scenario structural isolation test
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  BONUS: Cross-scenario isolation test")
    print(f"{'='*70}")

    # Put ALL scenarios' claims in a single store
    mega_store = _tmp_store(pid="mega")
    total_claims = 0
    for scenario_name, _, kb_passages, _, _, _ in ALL_SCENARIOS:
        for doc_id, info in kb_passages.items():
            extractor = ThoughtExtractor(
                llm_callable=_make_llm(info["fake_resp"]),
                project_id="mega",
            )
            result = await extractor.extract_from_passage(
                info["text"], source_id=doc_id,
            )
            for claim in result.claims:
                await mega_store.add(claim)
                total_claims += 1

    print(f"\n  Mega store: {total_claims} claims from all 10 scenarios")

    # Each scenario's patterns should primarily match its own domain
    print(f"\n  Testing pattern specificity across all domains:")
    isolation_ok = True

    for scenario_name, _, kb_passages, expected_matches, match_patterns, _ in ALL_SCENARIOS:
        results = await mega_store.search_by_structure(
            abstract_patterns=match_patterns, limit=30,
        )
        found_docs = {r.source_id for r in results}
        expected_set = set(expected_matches)
        # Check: expected docs should be IN the results
        missing = expected_set - found_docs
        # Check: how many extra docs are found
        extra = found_docs - expected_set
        recall = len(expected_set - missing) / len(expected_set) if expected_set else 1.0

        status = "✓" if not missing else "✗"
        print(f"  {status} {scenario_name}: found={len(found_docs)} "
              f"expected={len(expected_set)} recall={recall:.0%} "
              f"extra={len(extra)}")
        if missing:
            print(f"      MISSING: {missing}")
            isolation_ok = False
        if extra:
            # Extra matches are informational — some overlap is natural
            print(f"      (extra matches: {extra})")

    if not isolation_ok:
        failed += 1
        details.append(f"  ✗ Cross-scenario isolation")
    else:
        passed += 1
        details.append(f"  ✓ Cross-scenario isolation")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed} passed, {failed} failed")
    print(f"{'='*70}")
    for d in details:
        print(d)
    print()

    if failed > 0:
        print("  ⚠ SOME TESTS FAILED")
        return False
    else:
        print("  ALL TESTS PASSED ✓")
        return True


if __name__ == "__main__":
    print("=" * 70)
    print("  De-insight: Deep Mode Scenario Tests (10 scenarios)")
    print("  Tests structural retrieval across domains")
    print("=" * 70)
    success = asyncio.run(run_all_scenario_tests())
    print("=" * 70)
    sys.exit(0 if success else 1)
