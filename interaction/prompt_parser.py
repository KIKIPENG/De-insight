import re
from dataclasses import dataclass, field

@dataclass
class InteractiveBlock:
    type: str
    prompt: str
    choices: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0

def _choices(raw: str) -> list[str]:
    """從文字中解析選項列表（- 開頭的行）。"""
    return [l.lstrip('- ').strip() for l in raw.strip().splitlines()
            if l.strip().startswith('-') and len(l.strip()) > 1]

# ── 有 closing tag 的格式 ──
SELECT_CLOSED  = re.compile(r'<<SELECT[:：]\s*(.+?)>>(.*?)<</SELECT>>', re.DOTALL)
MULTI_CLOSED   = re.compile(r'<<MULTI[:：]\s*(.+?)>>(.*?)<</MULTI>>', re.DOTALL)
INPUT_CLOSED   = re.compile(r'<<INPUT[:：]\s*(.+?)>>\s*<</INPUT>>', re.DOTALL)
CONFIRM_CLOSED = re.compile(r'<<CONFIRM[:：]\s*(.+?)>>\s*<</CONFIRM>>', re.DOTALL)

# ── 無 closing tag 的單行格式（fallback）──
# <<SELECT: prompt\n- a\n- b\n- c>>
SELECT_INLINE  = re.compile(r'<<SELECT[:：]\s*(.+?)(?:\n((?:\s*-\s*.+\n?)+))\s*>>', re.DOTALL)
MULTI_INLINE   = re.compile(r'<<MULTI[:：]\s*(.+?)(?:\n((?:\s*-\s*.+\n?)+))\s*>>', re.DOTALL)
# <<INPUT: prompt>>
INPUT_INLINE   = re.compile(r'<<INPUT[:：]\s*(.+?)>>', re.DOTALL)
# <<CONFIRM: prompt>>
CONFIRM_INLINE = re.compile(r'<<CONFIRM[:：]\s*(.+?)>>', re.DOTALL)


def parse_interactive_blocks(text: str) -> tuple[str, list[InteractiveBlock]]:
    blocks: list[InteractiveBlock] = []
    matched_spans: list[tuple[int, int]] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(s <= start < e or s < end <= e for s, e in matched_spans)

    # 先嘗試有 closing tag 的格式（優先）
    for m in SELECT_CLOSED.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('select', m.group(1).strip(),
                _choices(m.group(2)), m.start(), m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in MULTI_CLOSED.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('multi', m.group(1).strip(),
                _choices(m.group(2)), m.start(), m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in INPUT_CLOSED.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('input', m.group(1).strip(),
                start=m.start(), end=m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in CONFIRM_CLOSED.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('confirm', m.group(1).strip(),
                start=m.start(), end=m.end()))
            matched_spans.append((m.start(), m.end()))

    # 再嘗試無 closing tag 的格式（fallback）
    for m in SELECT_INLINE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('select', m.group(1).strip(),
                _choices(m.group(2)), m.start(), m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in MULTI_INLINE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('multi', m.group(1).strip(),
                _choices(m.group(2)), m.start(), m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in INPUT_INLINE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('input', m.group(1).strip(),
                start=m.start(), end=m.end()))
            matched_spans.append((m.start(), m.end()))

    for m in CONFIRM_INLINE.finditer(text):
        if not _overlaps(m.start(), m.end()):
            blocks.append(InteractiveBlock('confirm', m.group(1).strip(),
                start=m.start(), end=m.end()))
            matched_spans.append((m.start(), m.end()))

    blocks.sort(key=lambda b: b.start)

    # 清除所有匹配的標記
    clean = text
    for start, end in sorted(matched_spans, reverse=True):
        clean = clean[:start] + clean[end:]

    return clean.strip(), blocks
