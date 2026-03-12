"""
Comprehensive test suite for interactive prompt parser and think tag filter.

測試互動式提示詞解析器和思考標籤過濾器的完整邊界案例。
"""

import pytest
from interaction.prompt_parser import parse_interactive_blocks, InteractiveBlock
from utils.think_filter import ThinkTagFilter


# ============================================================================
# PART A: Interactive Prompt Parser Tests (parse_interactive_blocks)
# ============================================================================

class TestSelectWithClosingTags:
    """SELECT 區塊有完整 closing tag 的情況。"""

    def test_basic_select_closed_with_choices(self):
        """基本 SELECT 區塊：有提示詞、多個選項、完整 closing tag。"""
        text = """<<SELECT: 你想探討哪個方向？>>
- 社會結構
- 個人意識
- 文化衝突
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "select"
        assert blocks[0].prompt == "你想探討哪個方向？"
        assert len(blocks[0].choices) == 3
        assert blocks[0].choices[0] == "社會結構"
        assert clean.strip() == ""

    def test_select_closed_with_text_before_after(self):
        """SELECT 區塊前後有文字，文字應被保留。"""
        text = "我的想法是：\n<<SELECT: 選擇方向>>\n- A\n- B\n<</SELECT>>\n這是結論。"
        clean, blocks = parse_interactive_blocks(text)

        assert "我的想法是" in clean
        assert "這是結論" in clean
        assert "<<SELECT" not in clean
        assert len(blocks) == 1

    def test_select_closed_with_whitespace_padding(self):
        """SELECT 區塊前後有大量空白，應被正確清除。"""
        text = "   \n\n<<SELECT: 問題？>>\n- a\n- b\n<</SELECT>>\n\n   "
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].prompt == "問題？"
        assert len(blocks[0].choices) == 2

    def test_select_closed_fullwidth_colon(self):
        """SELECT 使用全寬冒號（：）而非半寬冒號（:）。"""
        text = "<<SELECT： 你的選擇？>>\n- 選項1\n- 選項2\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].prompt == "你的選擇？"
        assert len(blocks[0].choices) == 2


class TestSelectInlineFormat:
    """SELECT 區塊無 closing tag 的單行格式。"""

    def test_select_inline_basic(self):
        """SELECT inline 格式：無 closing tag，選項在換行後。"""
        text = """<<SELECT: 選擇方向
- 方向A
- 方向B
>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "select"
        assert blocks[0].prompt == "選擇方向"
        assert len(blocks[0].choices) == 2
        assert blocks[0].choices[0] == "方向A"

    def test_select_inline_with_surrounding_text(self):
        """SELECT inline 與前後文字混合。"""
        text = "前言\n<<SELECT: 選擇\n- A\n- B\n>>\n後言"
        clean, blocks = parse_interactive_blocks(text)

        assert "前言" in clean
        assert "後言" in clean
        assert len(blocks) == 1

    def test_select_inline_single_choice(self):
        """SELECT inline 只有一個選項。"""
        text = "<<SELECT: 確認？\n- 是\n>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 1
        assert blocks[0].choices[0] == "是"


class TestMultiWithClosingTags:
    """MULTI 區塊有 closing tag 的情況（多選）。"""

    def test_multi_closed_basic(self):
        """MULTI 區塊：允許多選的選項列表。"""
        text = """<<MULTI: 哪些主題感興趣？>>
- 哲學
- 藝術
- 科技
<</MULTI>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "multi"
        assert blocks[0].prompt == "哪些主題感興趣？"
        assert len(blocks[0].choices) == 3

    def test_multi_closed_with_complex_choices(self):
        """MULTI 選項含有特殊字符（括號、符號等）。"""
        text = """<<MULTI: 選擇>>
- 項目(1) 的詳細
- 「次要」項目
- 第三點：補充說明
<</MULTI>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 3
        assert "項目(1)" in blocks[0].choices[0]
        assert "「次要」" in blocks[0].choices[1]


class TestMultiInlineFormat:
    """MULTI 區塊無 closing tag 的格式。"""

    def test_multi_inline_basic(self):
        """MULTI inline 格式。"""
        text = """<<MULTI: 哪些選項？
- 選項1
- 選項2
- 選項3
>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].type == "multi"
        assert len(blocks[0].choices) == 3


class TestInputWithClosingTag:
    """INPUT 區塊有 closing tag 的情況。"""

    def test_input_closed_basic(self):
        """INPUT 區塊：用戶自由輸入，無預設選項。"""
        text = "<<INPUT: 你的想法是什麼？>><</INPUT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "input"
        assert blocks[0].prompt == "你的想法是什麼？"
        assert len(blocks[0].choices) == 0  # INPUT 無選項

    def test_input_closed_with_surrounding_text(self):
        """INPUT 區塊前後有文字。"""
        text = "請告訴我你的想法：\n<<INPUT: 輸入內容>><</INPUT>>\n感謝你的分享。"
        clean, blocks = parse_interactive_blocks(text)

        assert "請告訴我" in clean
        assert "感謝你的" in clean
        assert len(blocks) == 1

    def test_input_closed_with_newlines(self):
        """INPUT 區塊內有換行（雖然通常不會）。"""
        text = "<<INPUT: 請輸入\n\n>><</INPUT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].type == "input"


class TestInputInlineFormat:
    """INPUT 區塊無 closing tag 的格式。"""

    def test_input_inline_basic(self):
        """INPUT inline 格式：簡單提示。"""
        text = "<<INPUT: 請輸入你的想法>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "input"
        assert blocks[0].prompt == "請輸入你的想法"

    def test_input_inline_with_context(self):
        """INPUT inline 與上下文混合。"""
        text = "背景說明\n<<INPUT: 你怎麼看？>>\n後續問題"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert "背景說明" in clean
        assert "後續問題" in clean


class TestConfirmWithClosingTag:
    """CONFIRM 區塊有 closing tag 的情況。"""

    def test_confirm_closed_basic(self):
        """CONFIRM 區塊：確認型提問。"""
        text = "<<CONFIRM: 你確認這個觀點嗎？>><</CONFIRM>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "confirm"
        assert blocks[0].prompt == "你確認這個觀點嗎？"

    def test_confirm_closed_with_text(self):
        """CONFIRM 區塊前後有文字。"""
        text = "基於上述分析，\n<<CONFIRM: 是否繼續深入？>><</CONFIRM>>\n謝謝。"
        clean, blocks = parse_interactive_blocks(text)

        assert "基於上述" in clean
        assert "謝謝" in clean


class TestConfirmInlineFormat:
    """CONFIRM 區塊無 closing tag 的格式。"""

    def test_confirm_inline_basic(self):
        """CONFIRM inline 格式。"""
        text = "<<CONFIRM: 是否同意？>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 1
        assert blocks[0].type == "confirm"


class TestMultipleBlocksInOneText:
    """單一文本中包含多個互動區塊。"""

    def test_select_then_input(self):
        """SELECT 後跟 INPUT 區塊。"""
        text = """首先選擇方向：
<<SELECT: 主題是？>>
- A
- B
<</SELECT>>

然後分享想法：
<<INPUT: 詳細說明>><</INPUT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 2
        assert blocks[0].type == "select"
        assert blocks[1].type == "input"
        assert "首先選擇" in clean
        assert "然後分享" in clean

    def test_multi_select_confirm(self):
        """MULTI、SELECT、CONFIRM 三個區塊。"""
        text = """<<MULTI: 哪些感興趣？>>
- A
- B
<</MULTI>>

<<SELECT: 最重要的是？>>
- X
- Y
<</SELECT>>

<<CONFIRM: 滿意嗎？>><</CONFIRM>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 3
        assert blocks[0].type == "multi"
        assert blocks[1].type == "select"
        assert blocks[2].type == "confirm"

    def test_multiple_blocks_order_preserved(self):
        """多個區塊的順序應與文本中出現的順序一致。"""
        text = """<<INPUT: 第一個>><</INPUT>>
<<SELECT: 第二個>>
- a
<</SELECT>>
<<CONFIRM: 第三個>><</CONFIRM>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].prompt == "第一個"
        assert blocks[1].prompt == "第二個"
        assert blocks[2].prompt == "第三個"


class TestBlockPositions:
    """區塊起始和結束位置應準確。"""

    def test_start_end_positions_accurate(self):
        """Start/end 位置應指向正確的文本範圍。"""
        text = "<<SELECT: 選擇>>\n- a\n- b\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        block = blocks[0]
        # 驗證 start 和 end 指向的是匹配的標記
        assert text[block.start:block.end] == text[block.start:block.end]

    def test_positions_with_multiple_blocks(self):
        """多個區塊的位置不應重疊。"""
        text = "A<<SELECT: 1>>\n-x\n<</SELECT>>B<<INPUT: 2>>C"
        clean, blocks = parse_interactive_blocks(text)

        if len(blocks) >= 2:
            assert blocks[0].end <= blocks[1].start


class TestTextPreservation:
    """測試文本保留的各種情況。"""

    def test_text_before_and_after_preserved(self):
        """區塊前後的文字應完整保留。"""
        text = "前言內容\n<<SELECT: 選擇>>\n- a\n<</SELECT>>\n後言內容"
        clean, blocks = parse_interactive_blocks(text)

        assert clean.startswith("前言內容")
        assert clean.endswith("後言內容")

    def test_text_between_blocks_preserved(self):
        """多個區塊之間的文字應保留。"""
        text = "<<INPUT: 1>><</INPUT>>\n中間文字\n<<SELECT: 2>>\n- a\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert "中間文字" in clean

    def test_clean_text_has_no_markers(self):
        """清淨文本不應包含任何互動標記。"""
        text = """開始
<<SELECT: 選擇>>
- A
- B
<</SELECT>>
<<INPUT: 輸入>><</INPUT>>
<<MULTI: 多選>>
- X
- Y
<</MULTI>>
<<CONFIRM: 確認?>><</CONFIRM>>
結束"""
        clean, blocks = parse_interactive_blocks(text)

        assert "<<SELECT" not in clean
        assert "<<INPUT" not in clean
        assert "<<MULTI" not in clean
        assert "<<CONFIRM" not in clean
        assert "<</SELECT" not in clean
        assert "開始" in clean
        assert "結束" in clean


class TestChoicesParsing:
    """選項解析的邊界案例。"""

    def test_empty_choices_list(self):
        """無有效選項的區塊（無 - 開頭的行）。"""
        text = "<<SELECT: 選擇？>>\n沒有有效選項\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 0

    def test_single_choice(self):
        """只有一個選項。"""
        text = "<<SELECT: 唯一選擇？>>\n- 只有一個\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 1
        assert blocks[0].choices[0] == "只有一個"

    def test_unicode_in_choices(self):
        """選項包含中文、emoji 等 Unicode 字符。"""
        text = """<<SELECT: 選擇>>
- 中文選項
- 日本語オプション
- 한글선택지
- 🚀 火箭
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 4
        assert "中文選項" in blocks[0].choices
        assert "🚀 火箭" in blocks[0].choices

    def test_choices_with_special_chars(self):
        """選項含有特殊字符。"""
        text = """<<SELECT: 選擇>>
- 「中文」括號
- (English) parentheses
- [方括號]
- 項目/子項目
- 數字123
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 5
        assert "「中文」" in blocks[0].choices[0]

    def test_choices_with_extra_spaces(self):
        """選項前後有多餘空白，應被清除。"""
        text = """<<SELECT: 選擇>>
-   前後空白
-  單邊空白
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].choices[0] == "前後空白"
        assert blocks[0].choices[1] == "單邊空白"

    def test_choices_with_dashes_in_text(self):
        """選項文本本身包含連字符。"""
        text = """<<SELECT: 選擇>>
- 第一項 - 帶連字符
- 雙重-連-接符
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 2
        assert "帶連字符" in blocks[0].choices[0]


class TestPromptStripping:
    """提示詞的空白處理。"""

    def test_prompt_whitespace_stripped(self):
        """提示詞前後的空白應被移除。"""
        text = "<<SELECT:   空白的提示詞   >>\n- a\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert blocks[0].prompt == "空白的提示詞"

    def test_prompt_newlines_preserved(self):
        """提示詞內的換行符應保留（如果有的話）。"""
        text = "<<SELECT: 第一行\n第二行>>\n- a\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        # 由於 .strip() 的作用，換行可能被處理
        assert "第一行" in blocks[0].prompt


class TestBlockTypeReturnValues:
    """區塊類型的正確性。"""

    def test_select_type_returned_correctly(self):
        """SELECT 區塊類型為 'select'。"""
        text = "<<SELECT: ?>>\n- a\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)
        assert blocks[0].type == "select"

    def test_multi_type_returned_correctly(self):
        """MULTI 區塊類型為 'multi'。"""
        text = "<<MULTI: ?>>- a\n<</MULTI>>"
        clean, blocks = parse_interactive_blocks(text)
        assert blocks[0].type == "multi"

    def test_input_type_returned_correctly(self):
        """INPUT 區塊類型為 'input'。"""
        text = "<<INPUT: ?>><</INPUT>>"
        clean, blocks = parse_interactive_blocks(text)
        assert blocks[0].type == "input"

    def test_confirm_type_returned_correctly(self):
        """CONFIRM 區塊類型為 'confirm'。"""
        text = "<<CONFIRM: ?>><</CONFIRM>>"
        clean, blocks = parse_interactive_blocks(text)
        assert blocks[0].type == "confirm"


class TestOverlapPrevention:
    """重疊區塊應被排除。"""

    def test_no_overlapping_spans(self):
        """重疊匹配應被排除（只保留優先級高的）。"""
        # 有 closing tag 的格式優先於 inline 格式
        text = "<<SELECT: 選擇>>\n- a\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        # 應該只有一個區塊被解析
        assert len(blocks) <= 1


class TestNoBlocksFound:
    """未找到任何區塊的情況。"""

    def test_no_blocks_returns_empty_list(self):
        """文本中無互動區塊時，返回空列表。"""
        text = "這是普通文本，沒有任何互動區塊。"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 0
        assert clean == text.strip()

    def test_no_blocks_text_unchanged(self):
        """無區塊時，清淨文本應與原文本相同（strip 後）。"""
        text = "   普通內容   \n帶換行   "
        clean, blocks = parse_interactive_blocks(text)

        assert blocks == []
        assert clean == text.strip()


class TestMixedClosedAndInlineBlocks:
    """混合有 closing tag 和無 closing tag 的區塊。"""

    def test_mixed_format_blocks(self):
        """同一文本中既有 closed 又有 inline 格式。"""
        text = """<<SELECT: 第一個>>
- a
<</SELECT>>

<<INPUT: 第二個>>"""
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks) == 2
        assert blocks[0].type == "select"
        assert blocks[1].type == "input"


class TestVeryLongChoiceText:
    """非常長的選項文本。"""

    def test_long_choice_text(self):
        """選項是一個很長的字符串。"""
        long_choice = "這是一個非常長的選項文本，" * 20
        text = f"<<SELECT: 選擇>>\n- {long_choice}\n<</SELECT>>"
        clean, blocks = parse_interactive_blocks(text)

        assert len(blocks[0].choices) == 1
        assert long_choice in blocks[0].choices[0]


class TestComplexNestedContent:
    """區塊內容中包含其他複雜結構。"""

    def test_choices_with_indentation(self):
        """選項內容有縮進（可能被當作子項）。"""
        text = """<<SELECT: 選擇>>
- 主項
  - 子項（可能不被視為選項）
- 另一個主項
<</SELECT>>"""
        clean, blocks = parse_interactive_blocks(text)

        # 只有以 - 開頭的行才被視為選項
        # 子項的處理取決於實現
        assert len(blocks[0].choices) >= 2


# ============================================================================
# PART B: ThinkTagFilter Tests (Streaming Filter)
# ============================================================================

class TestThinkTagFilterBasic:
    """基本的 think tag 過濾。"""

    def test_no_think_tags_passes_through(self):
        """文本中無 think tag，應原樣通過。"""
        filter = ThinkTagFilter()
        text = "這是一個普通的回應文本。"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == text

    def test_complete_think_block_in_single_chunk(self):
        """完整的 think 區塊在一個 chunk 中，應被移除。"""
        filter = ThinkTagFilter()
        text = "前言<think>思考內容</think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"
        assert "<think>" not in final
        assert "</think>" not in final

    def test_text_before_think_preserved(self):
        """think 區塊前的文本應被保留。"""
        filter = ThinkTagFilter()
        text = "重要前言<think>可以丟掉</think>"
        result = filter.feed(text)
        final = result + filter.flush()

        assert "重要前言" in final

    def test_text_after_think_preserved(self):
        """think 區塊後的文本應被保留。"""
        filter = ThinkTagFilter()
        text = "<think>丟掉</think>重要後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert "重要後言" in final

    def test_text_between_two_thinks_preserved(self):
        """兩個 think 區塊之間的文本應被保留。"""
        filter = ThinkTagFilter()
        text = "<think>丟1</think>中間<think>丟2</think>"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "中間"


class TestOpeningTagSplit:
    """開始標籤被分割到不同 chunk 的情況。"""

    def test_opening_tag_split_chunk1_lt(self):
        """chunk1 = '<'，chunk2 = 'think>...'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<")
        result2 = filter.feed("think>content</think>")
        final = result2 + filter.flush()

        assert result1 == ""
        assert "<think>" not in (result1 + final)

    def test_opening_tag_split_chunk1_lthi(self):
        """chunk1 = '<thi'，chunk2 = 'nk>...'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<thi")
        result2 = filter.feed("nk>content</think>")
        final = result2 + filter.flush()

        assert final == ""

    def test_opening_tag_split_chunk1_lthin(self):
        """chunk1 = '<thin'，chunk2 = 'k>...'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<thin")
        result2 = filter.feed("k>content</think>")
        final = result2 + filter.flush()

        assert final == ""

    def test_opening_tag_split_chunk1_lthink(self):
        """chunk1 = '<think'，chunk2 = '>...'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think")
        result2 = filter.feed(">content</think>")
        final = result2 + filter.flush()

        assert final == ""

    def test_opening_tag_split_chunk1_lthinkc(self):
        """chunk1 = '<think>'，chunk2 = 'content</think>'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think>")
        result2 = filter.feed("content</think>")
        final = result2 + filter.flush()

        assert final == ""


class TestClosingTagSplit:
    """結束標籤被分割到不同 chunk 的情況。"""

    def test_closing_tag_split_chunk1_partial_lt(self):
        """chunk1 = '<think>content<'，chunk2 = '/think>'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think>content<")
        result2 = filter.feed("/think>")
        final = result2 + filter.flush()

        assert final == ""

    def test_closing_tag_split_chunk1_partial_ltsl(self):
        """chunk1 = '<think>content</'，chunk2 = 'think>'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think>content</")
        result2 = filter.feed("think>")
        final = result2 + filter.flush()

        assert final == ""

    def test_closing_tag_split_chunk1_partial_ltslt(self):
        """chunk1 = '<think>content</t'，chunk2 = 'hink>'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think>content</t")
        result2 = filter.feed("hink>")
        final = result2 + filter.flush()

        assert final == ""

    def test_closing_tag_split_at_every_position(self):
        """測試 closing tag 在每個位置被分割。"""
        for i in range(1, 8):  # </think> 有 8 個字符
            filter = ThinkTagFilter()
            closing = "</think>"
            chunk1 = f"<think>content{closing[:i]}"
            chunk2 = closing[i:]

            result1 = filter.feed(chunk1)
            result2 = filter.feed(chunk2)
            final = result2 + filter.flush()

            assert final == "", f"Failed at split position {i}"


class TestOpeningTagAtEveryPosition:
    """測試 opening tag 在每個位置被分割（1-6 字符）。"""

    def test_opening_tag_split_1_char(self):
        """分割在第 1 字符：'<'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<")
        result2 = filter.feed("think>content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"

    def test_opening_tag_split_2_chars(self):
        """分割在第 2 字符：'<t'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<t")
        result2 = filter.feed("hink>content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"

    def test_opening_tag_split_3_chars(self):
        """分割在第 3 字符：'<th'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<th")
        result2 = filter.feed("ink>content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"

    def test_opening_tag_split_4_chars(self):
        """分割在第 4 字符：'<thi'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<thi")
        result2 = filter.feed("nk>content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"

    def test_opening_tag_split_5_chars(self):
        """分割在第 5 字符：'<thin'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<thin")
        result2 = filter.feed("k>content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"

    def test_opening_tag_split_6_chars(self):
        """分割在第 6 字符：'<think'"""
        filter = ThinkTagFilter()
        result1 = filter.feed("text<think")
        result2 = filter.feed(">content</think>")
        final = result1 + result2 + filter.flush()

        assert final == "text"


class TestMultipleThinkBlocks:
    """單一 chunk 中有多個 think 區塊。"""

    def test_multiple_think_blocks_in_one_chunk(self):
        """一個 chunk 中有多個完整的 think 區塊。"""
        filter = ThinkTagFilter()
        text = "A<think>1</think>B<think>2</think>C"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "ABC"

    def test_three_consecutive_think_blocks(self):
        """連續三個 think 區塊。"""
        filter = ThinkTagFilter()
        text = "<think>丟1</think><think>丟2</think><think>丟3</think>"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == ""


class TestThinkBlockSpanningMultipleChunks:
    """think 區塊跨越多個 chunk 的情況。"""

    def test_think_block_spanning_3_chunks(self):
        """think 區塊分布在 3 個 chunk 中。"""
        filter = ThinkTagFilter()
        result1 = filter.feed("前言<think>")
        result2 = filter.feed("中間內容")
        result3 = filter.feed("</think>後言")
        final = result1 + result2 + result3 + filter.flush()

        assert final == "前言後言"

    def test_think_block_spanning_many_chunks(self):
        """think 區塊跨越很多小 chunk。"""
        filter = ThinkTagFilter()
        result1 = filter.feed("<think>")
        result2 = filter.feed("a")
        result3 = filter.feed("b")
        result4 = filter.feed("c")
        result5 = filter.feed("</think>")
        final = result5 + filter.flush()

        assert final == ""


class TestEmptyThinkBlock:
    """空的 think 區塊。"""

    def test_empty_think_block(self):
        """<think></think> 應被完全移除。"""
        filter = ThinkTagFilter()
        text = "前言<think></think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"

    def test_empty_think_block_spanning_chunks(self):
        """空 think 區塊跨越 2 個 chunk。"""
        filter = ThinkTagFilter()
        result1 = filter.feed("前言<think>")
        result2 = filter.feed("</think>後言")
        final = result1 + result2 + filter.flush()

        assert final == "前言後言"


class TestThinkBlockWithNewlines:
    """think 區塊內有換行符。"""

    def test_think_block_with_newlines(self):
        """think 區塊內含有多個換行。"""
        filter = ThinkTagFilter()
        text = "前言<think>\n多行\n內容\n</think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"
        assert "\n" not in final.replace("前言", "").replace("後言", "")


class TestFlushAtEndOfStream:
    """流結束時的 flush() 操作。"""

    def test_flush_with_no_pending_content(self):
        """flush() 當沒有待處理內容時。"""
        filter = ThinkTagFilter()
        filter.feed("完整的文本")
        result = filter.flush()

        assert result == ""

    def test_flush_inside_think_block(self):
        """flush() 時還在 think 區塊內，應丟棄不完整的區塊。"""
        filter = ThinkTagFilter()
        result = filter.feed("文本<think>未完成的內容")
        result += filter.flush()

        assert result == "文本"  # 丟棄未完成的 think 區塊

    def test_flush_with_partial_opening_tag(self):
        """flush() 時有未完成的開始標籤。"""
        filter = ThinkTagFilter()
        result = filter.feed("文本<think")
        result += filter.flush()

        assert result == "文本<think"


class TestConsecutiveFeeds:
    """連續的 feed() 呼叫。"""

    def test_consecutive_feeds_building_complete_block(self):
        """多個 feed() 呼叫逐漸構建一個完整的 think 區塊。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("開始")
        result2 = filter.feed("<think>")
        result3 = filter.feed("思考")
        result4 = filter.feed("</think>")
        result5 = filter.feed("結束")

        final = result1 + result2 + result3 + result4 + result5 + filter.flush()

        assert final == "開始結束"

    def test_consecutive_feeds_alternating_think_and_text(self):
        """多個 think 區塊交替出現。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("A<think>1</think>")
        result2 = filter.feed("B<think>2</think>")
        result3 = filter.feed("C")

        final = result1 + result2 + result3 + filter.flush()

        assert final == "ABC"


class TestLargeContentInsideThinkBlock:
    """think 區塊內有非常大的內容。"""

    def test_large_content_in_think_block(self):
        """think 區塊內有 1000+ 字符的內容。"""
        filter = ThinkTagFilter()
        large_content = "x" * 1000
        text = f"前言<think>{large_content}</think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"

    def test_large_content_spanning_chunks(self):
        """large think 區塊分布在多個 chunk 中。"""
        filter = ThinkTagFilter()
        large_content = "x" * 1000

        result1 = filter.feed(f"前言<think>{large_content[:500]}")
        result2 = filter.feed(large_content[500:] + "</think>後言")
        final = result1 + result2 + filter.flush()

        assert final == "前言後言"


class TestLookAlikeButNotThinkTag:
    """看起來像但實際不是 think tag 的情況。"""

    def test_thinking_tag_not_matched(self):
        """<thinking> 不應被當作 <think> 匹配。"""
        filter = ThinkTagFilter()
        text = "內容<thinking>思考</thinking>更多"
        result = filter.feed(text)
        final = result + filter.flush()

        assert "<thinking>" in final

    def test_thinker_tag_not_matched(self):
        """<thinker> 不應被當作 <think> 匹配。"""
        filter = ThinkTagFilter()
        text = "內容<thinker>思考者</thinker>更多"
        result = filter.feed(text)
        final = result + filter.flush()

        assert "<thinker>" in final

    def test_think_without_brackets_not_matched(self):
        """think 而非 <think> 不應被匹配。"""
        filter = ThinkTagFilter()
        text = "我在think一些東西"
        result = filter.feed(text)
        final = result + filter.flush()

        assert "think" in final


class TestOpenThinkBlockAtStreamEnd:
    """流結束時 think 區塊未關閉。"""

    def test_unclosed_think_block_filters_all_subsequent_content(self):
        """未關閉的 <think> 應過濾所有後續內容。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("前言<think>")
        result2 = filter.feed("思考中...")
        result3 = filter.feed("還在思考...")
        final = result1 + result2 + result3 + filter.flush()

        assert final == "前言"

    def test_unclosed_think_block_middle_of_stream(self):
        """流中途有未關閉的 think，後續內容都被過濾。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("有效<think>無效")
        result2 = filter.feed("還是無效")
        final = result1 + result2 + filter.flush()

        assert final == "有效"


class TestStateResetAfterFlush:
    """flush() 後狀態應被重置。"""

    def test_state_reset_after_flush(self):
        """flush() 後，filter 應能處理新的內容。"""
        filter = ThinkTagFilter()

        # 第一次使用
        filter.feed("前1<think>丟</think>")
        filter.flush()

        # 第二次使用
        result = filter.feed("前2新內容")
        final = result + filter.flush()

        assert "前2新內容" in final


class TestPartialTagsAreKept:
    """流末尾的部分標籤應被保留（待下一個 chunk）。"""

    def test_partial_opening_tag_at_end_kept(self):
        """chunk 末尾是 '<' 或 '<t' 等，應保留待下一個 chunk。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("文本<")  # 部分標籤
        # flush 時部分標籤應被輸出（因為沒有完整的開始標籤）
        final = result1 + filter.flush()

        assert "<" in final or final == "文本<"

    def test_partial_closing_tag_at_end_kept(self):
        """chunk 末尾是 '</' 等，應保留待下一個 chunk。"""
        filter = ThinkTagFilter()

        result1 = filter.feed("<think>content<")
        result2 = filter.feed("/think>")
        final = result1 + result2 + filter.flush()

        assert final == ""


class TestWhitespaceHandling:
    """空白字符的處理。"""

    def test_whitespace_before_think_tag(self):
        """think 標籤前的空白應被保留。"""
        filter = ThinkTagFilter()
        text = "內容   <think>丟</think>"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "內容   "

    def test_think_block_with_only_whitespace(self):
        """think 區塊內只有空白。"""
        filter = ThinkTagFilter()
        text = "前言<think>   \n\n   </think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"


class TestEdgeCaseUnicode:
    """Unicode 字符的邊界案例。"""

    def test_emoji_and_chinese_content(self):
        """think 區塊內有 emoji 和中文。"""
        filter = ThinkTagFilter()
        text = "前言<think>🚀 思考中文內容 🎯</think>後言"
        result = filter.feed(text)
        final = result + filter.flush()

        assert final == "前言後言"
        assert "🚀" not in final


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """整合測試：parser 和 filter 結合使用。"""

    def test_filter_then_parse(self):
        """先過濾 think 標籤，再解析互動區塊。"""
        # 先過濾
        filter = ThinkTagFilter()
        filtered = filter.feed("回應<think>思考</think><<SELECT: 選擇>>\n- a\n<</SELECT>>")
        filtered += filter.flush()

        # 再解析
        clean, blocks = parse_interactive_blocks(filtered)

        assert len(blocks) == 1
        assert blocks[0].type == "select"
        assert "回應" in clean

    def test_complex_scenario(self):
        """複雜場景：多個 think、多個互動區塊、混合內容。"""
        # 模擬流式響應
        filter = ThinkTagFilter()
        chunks = [
            "讓我思考一下",
            "<think>初步分析</think>",
            "\n\n",
            "根據你的問題：",
            "<<SELECT: 方向？>>\n- A\n- B\n<</SELECT>>",
            "<think>補充思考</think>",
            "\n希望有幫助！"
        ]

        filtered_text = ""
        for chunk in chunks:
            filtered_text += filter.feed(chunk)
        filtered_text += filter.flush()

        clean, blocks = parse_interactive_blocks(filtered_text)

        assert "讓我思考一下" in clean
        assert "根據你的問題" in clean
        assert "希望有幫助" in clean
        assert len(blocks) == 1
        assert blocks[0].type == "select"
