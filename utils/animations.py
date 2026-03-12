"""ASCII 動畫幀定義與 AnimatedStatic widget。

色系與 DE_INSIGHT_THEME 一致：
  主色 #d4a27a（琥珀）  次色 #8b949e（灰）  暗色 #6e7681 / #484f58
"""

from __future__ import annotations

from textual.timer import Timer
from textual.widgets import Static


# ── 歡迎畫面：鑽石呼吸動畫 ─────────────────────────────────

WELCOME_FRAMES = [
    # 1 — 最小
    (
        "          [#6e7681]◇[/]\n"
        "         [#6e7681]╱ ╲[/]\n"
        "        [#6e7681]╱   ╲[/]\n"
        "       [#8b949e]◇[/] [#d4a27a]📖[/] [#8b949e]◇[/]\n"
        "        [#6e7681]╲   ╱[/]\n"
        "         [#6e7681]╲ ╱[/]\n"
        "          [#6e7681]◇[/]"
    ),
    # 2
    (
        "          [#8b949e]◇[/]\n"
        "         [#8b949e]╱ ╲[/]\n"
        "        [#8b949e]╱   ╲[/]\n"
        "       [#d4a27a]◇[/] [#d4a27a]📖[/] [#d4a27a]◇[/]\n"
        "        [#8b949e]╲   ╱[/]\n"
        "         [#8b949e]╲ ╱[/]\n"
        "          [#8b949e]◇[/]"
    ),
    # 3
    (
        "           [#d4a27a]◇[/]\n"
        "          [#c4925a]╱ ╲[/]\n"
        "         [#b88a50]╱   ╲[/]\n"
        "        [#d4a27a]◇[/]  [#d4a27a]📖[/]  [#d4a27a]◇[/]\n"
        "         [#b88a50]╲   ╱[/]\n"
        "          [#c4925a]╲ ╱[/]\n"
        "           [#d4a27a]◇[/]"
    ),
    # 4 — 最大
    (
        "            [#d4a27a]◇[/]\n"
        "          [#d4a27a]╱   ╲[/]\n"
        "         [#c4925a]╱     ╲[/]\n"
        "        [#d4a27a]◇[/]   [#d4a27a]📖[/]   [#d4a27a]◇[/]\n"
        "         [#c4925a]╲     ╱[/]\n"
        "          [#d4a27a]╲   ╱[/]\n"
        "            [#d4a27a]◇[/]"
    ),
    # 5 — 回縮
    (
        "           [#d4a27a]◇[/]\n"
        "          [#c4925a]╱ ╲[/]\n"
        "         [#b88a50]╱   ╲[/]\n"
        "        [#d4a27a]◇[/]  [#d4a27a]📖[/]  [#d4a27a]◇[/]\n"
        "         [#b88a50]╲   ╱[/]\n"
        "          [#c4925a]╲ ╱[/]\n"
        "           [#d4a27a]◇[/]"
    ),
    # 6
    (
        "          [#8b949e]◇[/]\n"
        "         [#8b949e]╱ ╲[/]\n"
        "        [#8b949e]╱   ╲[/]\n"
        "       [#d4a27a]◇[/] [#d4a27a]📖[/] [#d4a27a]◇[/]\n"
        "        [#8b949e]╲   ╱[/]\n"
        "         [#8b949e]╲ ╱[/]\n"
        "          [#8b949e]◇[/]"
    ),
    # 7 — 回到最暗
    (
        "          [#6e7681]◇[/]\n"
        "         [#6e7681]╱ ╲[/]\n"
        "        [#6e7681]╱   ╲[/]\n"
        "       [#8b949e]◇[/] [#d4a27a]📖[/] [#8b949e]◇[/]\n"
        "        [#6e7681]╲   ╱[/]\n"
        "         [#6e7681]╲ ╱[/]\n"
        "          [#6e7681]◇[/]"
    ),
]

# ── 思考中動畫：氣泡浮出與消散 ────────────────────────────

THINK_FRAMES = [
    "[#484f58]◇[/]                  ",
    "[#6e7681]◇[/] [#484f58]·[/]               ",
    "[#8b949e]◇[/] [#6e7681]·[/] [#484f58]·[/]            ",
    "[#d4a27a]◇[/] [#8b949e]·[/] [#6e7681]·[/] [#484f58]·[/]         ",
    "[#d4a27a]◇[/] [#d4a27a]·[/] [#8b949e]·[/] [#6e7681]·[/] [#484f58]·[/]      ",
    "[#d4a27a]◇[/] [#8b949e]·[/] [#6e7681]·[/] [#484f58]·[/]         ",
    "[#8b949e]◇[/] [#6e7681]·[/] [#484f58]·[/]            ",
    "[#6e7681]◇[/] [#484f58]·[/]               ",
    "[#484f58]◇[/]                  ",
]

# ── 記憶發現動畫：逐行浮現 ─────────────────────────────────

MEMORY_FRAMES = [
    # 1 — 空
    "                              ",
    # 2 — 燈泡出現
    "          [#d4a27a]💡[/]",
    # 3 — 上框線
    (
        "          [#d4a27a]💡[/]\n"
        "        [#484f58]╭─────╮[/]"
    ),
    # 4 — 框體 + 文字
    (
        "          [#d4a27a]💡[/]\n"
        "        [#6e7681]╭─────╮[/]\n"
        "        [#6e7681]│[/]     [#6e7681]│[/]  [#8b949e]發現了一個想法碎片[/]"
    ),
    # 5 — 完整框（最亮）
    (
        "          [#d4a27a]💡[/]\n"
        "        [#8b949e]╭─────╮[/]\n"
        "        [#8b949e]│[/]     [#8b949e]│[/]  [#d4a27a]發現了一個想法碎片[/]\n"
        "        [#8b949e]╰─────╯[/]"
    ),
    # 6 — 持續顯示
    (
        "          [#d4a27a]💡[/]\n"
        "        [#8b949e]╭─────╮[/]\n"
        "        [#8b949e]│[/]     [#8b949e]│[/]  [#d4a27a]發現了一個想法碎片[/]\n"
        "        [#8b949e]╰─────╯[/]"
    ),
]

# ── RAG 檢索動畫 ──────────────────────────────────────────

RAG_SEARCH_FRAMES = [
    "[#484f58]◇[/] [#6e7681]翻閱中[/]      ",
    "[#6e7681]◇[/]  [#6e7681]翻閱中[/] [#484f58].[/]   ",
    "[#8b949e]◇[/]   [#8b949e]翻閱中[/] [#6e7681].[/] [#484f58].[/]",
    "[#d4a27a]◇[/]    [#d4a27a]比對中[/] [#8b949e].[/] [#6e7681].[/]",
    "[#8b949e]◇[/]   [#8b949e]比對中[/] [#6e7681].[/] [#484f58].[/]",
    "[#6e7681]◇[/]  [#6e7681]整理中[/] [#484f58].[/]   ",
    "[#484f58]◇[/] [#6e7681]整理中[/]      ",
]


# ── AnimatedStatic widget ─────────────────────────────────

class AnimatedStatic(Static):
    """可播放 ASCII 幀動畫的 Static widget。

    Usage:
        anim = AnimatedStatic()
        anim.start(THINK_FRAMES, interval=0.35)
        # ...later...
        anim.stop()
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__("", *args, **kwargs)
        self._frames: list[str] = []
        self._interval: float = 0.5
        self._timer: Timer | None = None
        self._frame_idx: int = 0
        self._loop: bool = True
        self._on_complete: callable | None = None

    def start(
        self,
        frames: list[str],
        interval: float = 0.5,
        loop: bool = True,
        on_complete: callable | None = None,
    ) -> None:
        """開始播放動畫。

        Args:
            frames: Rich markup 幀列表
            interval: 每幀間隔（秒）
            loop: 是否循環播放
            on_complete: 非循環模式播放完畢的回呼
        """
        self.stop()
        self._frames = frames
        self._interval = interval
        self._loop = loop
        self._on_complete = on_complete
        self._frame_idx = 0
        if self._frames:
            self.update(self._frames[0])
            self._timer = self.set_interval(self._interval, self._next_frame)

    def stop(self) -> None:
        """停止動畫。"""
        if self._timer:
            self._timer.stop()
            self._timer = None

    def _next_frame(self) -> None:
        if not self._frames:
            return
        self._frame_idx += 1
        if self._frame_idx >= len(self._frames):
            if self._loop:
                self._frame_idx = 0
            else:
                self.stop()
                if self._on_complete:
                    self._on_complete()
                return
        self.update(self._frames[self._frame_idx])

    def on_unmount(self) -> None:
        """確保 widget 移除時停止計時器。"""
        self.stop()
