"""問題意識（Focus）── 讀寫與解析。"""

from __future__ import annotations

from pathlib import Path
import re


FOCUS_FIELDS = ["問題意識", "標籤", "作品形式", "目標", "限制"]

FIELD_ALIASES: dict[str, list[str]] = {
    "問題意識": ["問題意識", "核心問題", "core_question", "問題", "question", "主題"],
    "標籤": ["標籤", "tags", "tag", "keywords"],
    "作品形式": ["作品形式", "form", "format", "形式", "媒材"],
    "目標": ["目標", "goal", "goals", "目的", "aim"],
    "限制": ["限制", "邊界", "boundary", "limit", "不做", "排除"],
}

TEMPLATE = """\
---
問題意識: 
標籤: 
作品形式: 
目標: 
限制: 
---
"""


def focus_path(project_root: Path) -> Path:
    return project_root / "focus.md"


def load_focus(project_root: Path) -> dict[str, str]:
    """讀取 focus.md，回傳欄位字典。檔案不存在或解析失敗回傳空字串欄位。"""
    path = focus_path(project_root)
    if not path.exists():
        return {f: "" for f in FOCUS_FIELDS}
    try:
        parsed = _parse_frontmatter(path.read_text(encoding="utf-8"))
        return _normalize_fields_from_parsed(parsed)
    except Exception:
        return {f: "" for f in FOCUS_FIELDS}


def save_focus(project_root: Path, fields: dict[str, str]) -> None:
    """把欄位字典寫回 focus.md。"""
    path = focus_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["---"]
    for field in FOCUS_FIELDS:
        value = (fields.get(field, "") or "").strip()
        lines.append(f"{field}: {value}")
    lines.append("---")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def import_focus(raw: str) -> tuple[dict[str, str], dict[str, bool]]:
    """解析匯入 Markdown，回傳 (fields, matched)。"""
    parsed = _parse_frontmatter(raw)
    fields = _normalize_fields_from_parsed(parsed)
    matched = {f: bool((fields.get(f) or "").strip()) for f in FOCUS_FIELDS}

    return fields, matched


def to_prompt_block(fields: dict[str, str]) -> str:
    """把問題意識轉成 prompt 區塊，只輸出有值欄位。"""
    lines: list[str] = []
    for field in FOCUS_FIELDS:
        value = (fields.get(field, "") or "").strip()
        if value:
            lines.append(f"{field}：{value}")
    return "\n".join(lines)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """解析 YAML frontmatter，回傳 key-value 字典（均為字串）。"""
    result: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if match:
        block = match.group(1)
    else:
        result["問題意識"] = text.strip()
        return result

    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()

    return result


def _normalize_fields_from_parsed(parsed: dict[str, str]) -> dict[str, str]:
    out = {f: "" for f in FOCUS_FIELDS}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            value = parsed.get(alias, "")
            if value and value.strip():
                out[canonical] = value.strip()
                break
    return out
