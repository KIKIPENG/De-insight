"""測試 memory/thought_tracker.py 的工具函數。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import pytest
from unittest.mock import AsyncMock
from memory.thought_tracker import _clean_json, _text_overlap, extract_memories


class TestCleanJson:
    """測試 JSON 清理函數。"""

    def test_markdown_fence_json(self):
        input_text = '```json\n[{"type":"insight"}]\n```'
        expected = '[{"type":"insight"}]'
        assert _clean_json(input_text) == expected

    def test_markdown_fence_no_tag(self):
        input_text = '```\n[{"type":"insight"}]\n```'
        expected = '[{"type":"insight"}]'
        assert _clean_json(input_text) == expected

    def test_trailing_comma(self):
        input_text = '[{"type":"insight","content":"test"},]'
        expected = '[{"type":"insight","content":"test"}]'
        assert _clean_json(input_text) == expected

    def test_trailing_comma_object(self):
        input_text = '{"type":"insight","content":"test",}'
        expected = '{"type":"insight","content":"test"}'
        assert _clean_json(input_text) == expected

    def test_leading_text(self):
        input_text = '以下是記憶條目：\n[{"type":"insight"}]'
        expected = '[{"type":"insight"}]'
        assert _clean_json(input_text) == expected

    def test_single_quotes_conversion(self):
        input_text = "[{'type':'insight','content':'test'}]"
        expected = '[{"type":"insight","content":"test"}]'
        assert _clean_json(input_text) == expected

    def test_no_conversion_when_double_quotes_predominant(self):
        input_text = '[{"type":"insight","content":"it\'s fine"}]'
        result = _clean_json(input_text)
        assert "it's fine" in result or "it\\'s fine" in result

    def test_whitespace_handling(self):
        input_text = '  \n  [{"type":"insight"}]  \n  '
        expected = '[{"type":"insight"}]'
        assert _clean_json(input_text) == expected


class TestTextOverlap:
    """測試文字重疊率計算。"""

    def test_identical_text(self):
        assert _text_overlap("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert _text_overlap("abc", "xyz") == 0.0

    def test_partial_overlap(self):
        a = "包豪斯預設了功能的穩定性"
        b = "功能預設穩定"
        overlap = _text_overlap(a, b)
        assert 0.3 < overlap < 0.9

    def test_empty_string(self):
        assert _text_overlap("", "test") == 0.0
        assert _text_overlap("test", "") == 0.0
        assert _text_overlap("", "") == 0.0

    def test_case_insensitive(self):
        assert _text_overlap("ABC", "abc") == 1.0

    def test_paraphrase_vs_copy(self):
        original = "包豪斯那種功能決定形式的說法其實預設了功能是穩定的"
        paraphrase = "包豪斯預設功能穩定"
        copy = "包豪斯那種功能決定形式的說法其實預設了功能是穩定的"

        overlap_copy = _text_overlap(original, copy)
        overlap_paraphrase = _text_overlap(original, paraphrase)

        assert overlap_copy == 1.0
        assert overlap_paraphrase < 0.8


class TestExtractMemoriesRetry:
    """測試 extract_memories 的重試邏輯。"""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        mock_llm = AsyncMock(return_value='[{"type":"insight","content":"test insight here","topic":"設計史"}]')
        result = await extract_memories("一段超過40字的測試文字" * 5, mock_llm)
        assert len(result) == 1
        assert result[0]["type"] == "insight"
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_json_error(self):
        mock_llm = AsyncMock(side_effect=[
            "invalid json",
            "still bad{",
            '[{"type":"insight","content":"test insight here","topic":"設計史"}]'
        ])
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_then_give_up(self):
        mock_llm = AsyncMock(side_effect=[
            "bad json 1",
            "bad json 2",
            "bad json 3",
        ])
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert result == []
        assert mock_llm.call_count == 3

    @pytest.mark.asyncio
    async def test_filter_empty_content(self):
        mock_llm = AsyncMock(return_value='''
        [
            {"type":"insight","content":"","topic":"設計史"},
            {"type":"question","content":"有效問題超過五個字","topic":"美學"}
        ]
        ''')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1
        assert result[0]["content"] == "有效問題超過五個字"

    @pytest.mark.asyncio
    async def test_filter_short_content(self):
        mock_llm = AsyncMock(return_value='''
        [
            {"type":"insight","content":"好","topic":"設計史"},
            {"type":"question","content":"這是一個有效的問題","topic":"美學"}
        ]
        ''')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1
        assert result[0]["content"] == "這是一個有效的問題"

    @pytest.mark.asyncio
    async def test_filter_copied_text(self):
        user_text = "包豪斯那種功能決定形式的說法其實預設了功能是穩定的" * 2
        mock_llm = AsyncMock(return_value=f'''
        [
            {{"type":"insight","content":"{user_text}","topic":"設計史"}},
            {{"type":"insight","content":"包豪斯預設功能穩定","topic":"設計史"}}
        ]
        ''')
        result = await extract_memories(user_text, mock_llm)
        assert len(result) == 1
        assert "預設功能穩定" in result[0]["content"]

    @pytest.mark.asyncio
    async def test_filter_invalid_type(self):
        mock_llm = AsyncMock(return_value='''
        [
            {"type":"note","content":"筆記內容超過五字","topic":"設計史"},
            {"type":"insight","content":"洞見內容超過五字","topic":"美學"}
        ]
        ''')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1
        assert result[0]["type"] == "insight"


class TestLLMFormatVariations:
    """測試 LLM 各種輸出格式的解析。"""

    @pytest.mark.asyncio
    async def test_with_markdown_fence(self):
        mock_llm = AsyncMock(return_value='```json\n[{"type":"insight","content":"測試洞見超過五字","topic":"設計史"}]\n```')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_with_explanation_prefix(self):
        mock_llm = AsyncMock(return_value='以下是我從對話中抽取的記憶條目：\n\n[{"type":"insight","content":"測試洞見超過五字","topic":"設計史"}]')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_with_trailing_comma(self):
        mock_llm = AsyncMock(return_value='[\n    {"type":"insight","content":"測試洞見超過五字","topic":"設計史"},\n]')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_with_single_quotes(self):
        mock_llm = AsyncMock(return_value="[\n    {'type':'insight','content':'測試洞見超過五字','topic':'設計史'}\n]")
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_empty_array(self):
        mock_llm = AsyncMock(return_value='[]')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_items(self):
        mock_llm = AsyncMock(return_value='[\n    {"type":"insight","content":"洞見一超過五個字","topic":"設計史"},\n    {"type":"question","content":"問題一超過五個字","topic":"美學"},\n    {"type":"reaction","content":"反應一超過五個字","topic":"當代藝術"}\n]')
        result = await extract_memories("測試文字" * 10, mock_llm)
        assert len(result) == 3
        assert {r["type"] for r in result} == {"insight", "question", "reaction"}
