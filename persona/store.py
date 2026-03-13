"""Persona 儲存 — 全局批評視角的 CRUD。

每個 persona 是一個 JSON 檔案，存放在 data/personas/ 底下。
啟用中的 persona ID 清單存在 .env 的 ACTIVE_PERSONAS（逗號分隔）。

內建視角存放在 persona/builtins/（完整 movement JSON），
首次啟動時自動萃取並安裝到 data/personas/。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from paths import DATA_ROOT

log = logging.getLogger("de-insight.persona")

PERSONAS_DIR = DATA_ROOT / "personas"
BUILTINS_DIR = Path(__file__).resolve().parent / "builtins"

# prompt 注入的 token 軟上限（大約字數，非精確 token 計算）
_MAX_PERSONA_CHARS = 3000


def _ensure_dir() -> Path:
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
    return PERSONAS_DIR


# ── Builtins 自動安裝 ────────────────────────────────────────────


def _install_builtins() -> None:
    """從 persona/builtins/ 萃取並安裝內建批評視角。

    只在 data/personas/ 裡找不到對應檔案時才安裝，不會覆蓋。
    """
    if not BUILTINS_DIR.exists():
        return
    _ensure_dir()
    for f in sorted(BUILTINS_DIR.glob("*.json")):
        try:
            movement = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Failed to read builtin %s: %s", f.name, e)
            continue

        mid = movement.get("movement_id", f.stem)
        target = PERSONAS_DIR / f"{mid}.json"
        if target.exists():
            continue  # 已安裝，不覆蓋

        if not _is_valid_movement(movement):
            log.warning("Builtin %s is not a valid movement JSON, skip", f.name)
            continue

        persona_data = extract_persona_from_movement(movement)
        target.write_text(
            json.dumps(persona_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Builtin persona installed: %s → %s", f.name, target)


def _is_valid_movement(data: dict) -> bool:
    """檢查 dict 是否為有效的 movement JSON 格式。

    需要具備 movement_id 和 judge_persona_seed 兩個關鍵欄位。
    """
    if not isinstance(data, dict):
        return False
    if not data.get("movement_id"):
        return False
    judge = data.get("judge_persona_seed", {})
    if not isinstance(judge, dict):
        return False
    # judge_persona_seed 至少要有 personality
    if not judge.get("personality"):
        return False
    return True


# ── CRUD ─────────────────────────────────────────────────────────


def save_persona(persona_id: str, data: dict) -> Path:
    """儲存 persona JSON，回傳檔案路徑。"""
    _ensure_dir()
    path = PERSONAS_DIR / f"{persona_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Persona saved: %s", path)
    return path


def load_persona(persona_id: str) -> dict | None:
    """讀取單一 persona，不存在回傳 None。"""
    path = PERSONAS_DIR / f"{persona_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to load persona %s: %s", persona_id, e)
        return None


def list_personas() -> list[dict]:
    """列出所有已匯入的 persona，回傳 [{id, name_zh, name_en, domain}, ...]。

    首次呼叫時會自動安裝內建視角。
    """
    _install_builtins()
    result = []
    for f in sorted(PERSONAS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            result.append({
                "id": f.stem,
                "name_zh": data.get("name_zh", f.stem),
                "name_en": data.get("name_en", ""),
                "domain": data.get("domain", []),
            })
        except Exception:
            continue
    return result


def delete_persona(persona_id: str) -> bool:
    """刪除 persona，回傳是否成功。"""
    path = PERSONAS_DIR / f"{persona_id}.json"
    if path.exists():
        path.unlink()
        # 同時從啟用清單移除
        active = get_active_ids()
        if persona_id in active:
            active.remove(persona_id)
            set_active_ids(active)
        return True
    return False


# ── 啟用管理 ─────────────────────────────────────────────────────


def get_active_ids() -> list[str]:
    """取得目前啟用的 persona ID 清單。"""
    from config.service import get_config_service
    raw = get_config_service().snapshot().get("ACTIVE_PERSONAS", "")
    if not raw.strip():
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def set_active_ids(ids: list[str]) -> None:
    """設定啟用的 persona ID 清單。"""
    from config.service import get_config_service
    value = ",".join(ids) if ids else ""
    get_config_service().update_env({"ACTIVE_PERSONAS": value})


def toggle_persona(persona_id: str) -> bool:
    """切換 persona 啟用狀態，回傳切換後是否為啟用。"""
    active = get_active_ids()
    if persona_id in active:
        active.remove(persona_id)
        set_active_ids(active)
        return False
    else:
        active.append(persona_id)
        set_active_ids(active)
        return True


# ── Prompt 組裝 ──────────────────────────────────────────────────


def build_persona_prompt_block() -> str:
    """組裝所有啟用中 persona 的 prompt 注入區塊。

    無啟用 persona 時回傳空字串。
    啟用多個 persona 時依序組裝，超過字數上限後截斷。
    """
    active = get_active_ids()
    if not active:
        return ""

    blocks: list[str] = []
    total_chars = 0

    for pid in active:
        data = load_persona(pid)
        if not data:
            continue

        name = data.get("name_zh", pid)
        personality = data.get("personality", "")
        bias = data.get("evaluation_bias", "")
        blind_spots = data.get("blind_spots", "")
        typical_critique = data.get("typical_critique", "")

        # Fix #8: 加入 writing_style 的 voice rules
        voice_rules = data.get("judge_voice_rules", [])

        parts = [f"## {name}"]
        if personality:
            parts.append(personality)
        if bias:
            parts.append(f"你看事情的傾向：{bias}")
        if blind_spots:
            parts.append(f"你知道自己的盲點：{blind_spots}")
        if typical_critique:
            parts.append(f"你的典型提問方式：{typical_critique}")
        if voice_rules:
            rules_text = "\n".join(f"- {r}" for r in voice_rules[:5])
            parts.append(f"語氣規則：\n{rules_text}")

        block = "\n\n".join(parts)

        # Fix #6: token 上限保護
        if total_chars + len(block) > _MAX_PERSONA_CHARS and blocks:
            remaining = len(active) - len(blocks)
            log.info(
                "Persona prompt truncated: %d persona(s) skipped (char limit %d)",
                remaining, _MAX_PERSONA_CHARS,
            )
            break
        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return ""

    header = "# 這個對話的批評鏡頭\n\n"
    body = "\n\n---\n\n".join(blocks)
    footer = (
        "\n\n這些是觀看事物的角度，不是你的身份。"
        "\n你借用這些鏡頭觀察，但保留策展人自己的判斷。"
        "\n當某個視角的盲點正在影響你的判斷時，主動指出來。"
    )
    return header + body + footer


# ── Movement JSON 轉換 ──────────────────────────────────────────


def extract_persona_from_movement(movement: dict) -> dict:
    """從 movement JSON 抽取 persona 資料。

    只保留 judge_persona_seed + writing_style 中對 prompt 有用的部分。
    """
    judge = movement.get("judge_persona_seed", {})
    style = movement.get("writing_style", {})
    name = movement.get("name", {})
    domain = movement.get("domain", [])

    persona = {
        "name_zh": name.get("zh", movement.get("movement_id", "")),
        "name_en": name.get("en", ""),
        "domain": domain,
        # judge_persona_seed
        "personality": judge.get("personality", ""),
        "evaluation_bias": judge.get("evaluation_bias", ""),
        "blind_spots": judge.get("blind_spots", ""),
        "typical_critique": judge.get("typical_critique", ""),
        # writing_style（voice rules 用於進階控制）
        "collective_voice": style.get("collective_voice", ""),
        "judge_voice_rules": style.get("judge_voice_rules", []),
    }
    return persona


def extract_knowledge_text(movement: dict) -> str:
    """從 movement JSON 抽取知識性內容，轉為結構化 Markdown 文本。

    用於送入 LightRAG insert_text()。
    """
    name = movement.get("name", {})
    title = name.get("zh", "") or name.get("en", "")

    sections: list[str] = []

    # 標題
    sections.append(f"# {title}")
    if name.get("en"):
        sections.append(f"英文：{name['en']}")
    domain = movement.get("domain", [])
    if domain:
        sections.append(f"領域：{'、'.join(domain)}")

    # 時期與地理
    period = movement.get("period", {})
    if period:
        p_str = f"時期：{period.get('start', '?')}–{period.get('end') or '至今'}"
        if period.get("peak"):
            p_str += f"\n高峰期：{period['peak']}"
        sections.append(p_str)

    geography = movement.get("geography", [])
    if geography:
        sections.append("地理分布：\n" + "\n".join(f"- {g}" for g in geography))

    # 歷史脈絡
    ctx = movement.get("historical_context", {})
    if ctx.get("why_it_emerged"):
        sections.append(f"## 為何出現\n\n{ctx['why_it_emerged']}")
    if ctx.get("social_energy"):
        sections.append(f"## 社會能量\n\n{ctx['social_energy']}")
    if ctx.get("purpose"):
        sections.append(f"## 核心目標\n\n{ctx['purpose']}")
    timeline = ctx.get("key_timeline", [])
    if timeline:
        sections.append("## 關鍵時間線\n\n" + "\n".join(f"- {t}" for t in timeline))

    # 人物
    founders = movement.get("founders_and_masters", [])
    if founders:
        people_parts = ["## 代表人物"]
        for f in founders:
            fname = f.get("name", {})
            display = fname.get("zh", "") or fname.get("en", "")
            en = fname.get("en", "")
            if display and en and display != en:
                display = f"{display}（{en}）"
            people_parts.append(f"### {display}")
            if f.get("role"):
                people_parts.append(f.get("role"))
            works = f.get("key_works", [])
            if works:
                people_parts.append("代表作品：\n" + "\n".join(f"- {w}" for w in works))
            quotes = f.get("key_quotes", [])
            if quotes:
                for q in quotes:
                    text = q.get("text", "")
                    source = q.get("source", "")
                    meaning = q.get("meaning", "")
                    people_parts.append(f"> {text}")
                    if source:
                        people_parts.append(f"— {source}")
                    if meaning:
                        people_parts.append(meaning)
        sections.append("\n\n".join(people_parts))

    # 核心文獻
    texts = movement.get("core_texts", [])
    if texts:
        text_parts = ["## 核心文獻"]
        for t in texts:
            line = f"- {t.get('title', '')}（{t.get('author', '')}，{t.get('year', '')}）"
            if t.get("significance"):
                line += f"：{t['significance']}"
            text_parts.append(line)
        sections.append("\n".join(text_parts))

    # 核心精神
    spirit = movement.get("core_spirit", {})
    if spirit:
        spirit_parts = ["## 核心精神"]
        do_list = spirit.get("what_they_do", [])
        if do_list:
            spirit_parts.append("他們做什麼：\n" + "\n".join(f"- {d}" for d in do_list))
        refuse_list = spirit.get("what_they_refuse", [])
        if refuse_list:
            spirit_parts.append("他們拒絕什麼：\n" + "\n".join(f"- {r}" for r in refuse_list))
        sections.append("\n\n".join(spirit_parts))

    # 對立面
    opp = movement.get("opposition", {})
    if opp:
        opp_parts = ["## 對立與反對"]
        if opp.get("against_movements"):
            opp_parts.append("對抗的運動：\n" + "\n".join(f"- {m}" for m in opp["against_movements"]))
        if opp.get("against_values"):
            opp_parts.append("反對的價值：\n" + "\n".join(f"- {v}" for v in opp["against_values"]))
        if opp.get("societal_context"):
            opp_parts.append(f"社會脈絡：{opp['societal_context']}")
        sections.append("\n\n".join(opp_parts))

    # 解決的問題 / 製造的問題
    solved = movement.get("problems_solved", [])
    if solved:
        sections.append("## 解決的問題\n\n" + "\n".join(f"- {s}" for s in solved))
    created = movement.get("problems_created", [])
    if created:
        sections.append("## 製造的問題\n\n" + "\n".join(f"- {c}" for c in created))

    # 原創性分析
    orig = movement.get("originality_analysis", {})
    if orig:
        orig_parts = ["## 原創性分析"]
        if orig.get("breakthrough_idea"):
            orig_parts.append(f"突破性觀念：{orig['breakthrough_idea']}")
        if orig.get("what_was_truly_new"):
            orig_parts.append("真正的創新：\n" + "\n".join(f"- {n}" for n in orig["what_was_truly_new"]))
        if orig.get("what_was_borrowed"):
            orig_parts.append("借鑑的部分：\n" + "\n".join(f"- {b}" for b in orig["what_was_borrowed"]))
        if orig.get("creative_leap"):
            orig_parts.append(f"創造性飛躍：{orig['creative_leap']}")
        sections.append("\n\n".join(orig_parts))

    # 影響鏈
    chain = movement.get("influence_chain", {})
    if chain:
        chain_parts = ["## 影響鏈"]
        if chain.get("influenced_by"):
            chain_parts.append("受影響於：\n" + "\n".join(f"- {i}" for i in chain["influenced_by"]))
        if chain.get("influenced"):
            chain_parts.append("影響了：\n" + "\n".join(f"- {i}" for i in chain["influenced"]))
        sections.append("\n\n".join(chain_parts))

    return "\n\n".join(sections)
